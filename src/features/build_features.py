"""Build customer-level features at a given feature_end cutoff.

All features must be computed using ONLY transactions strictly before
feature_end, so we can join them safely to churn labels without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _basic_rfm(g: pd.DataFrame, ref: pd.Timestamp) -> pd.Series:
    """Recency / Frequency / Monetary at the customer level."""
    last = g["invoice_date"].max()
    first = g["invoice_date"].min()
    return pd.Series({
        "recency_days": (ref - last).days,
        "tenure_days": (ref - first).days,
        "frequency": g["invoice"].nunique(),
        "monetary": float(g["revenue"].sum()),
        "n_items": int(g["quantity"].sum()),
    })


def _behavioral(g: pd.DataFrame) -> pd.Series:
    """Basket size, purchase interval, diversity."""
    orders = g.groupby("invoice").agg(
        basket_revenue=("revenue", "sum"),
        basket_items=("quantity", "sum"),
        basket_skus=("stock_code", "nunique"),
        order_date=("invoice_date", "min"),
    )
    intervals = orders["order_date"].sort_values().diff().dt.days.dropna()
    return pd.Series({
        "avg_basket_revenue": float(orders["basket_revenue"].mean()),
        "avg_basket_items": float(orders["basket_items"].mean()),
        "avg_basket_skus": float(orders["basket_skus"].mean()),
        "mean_interval_days": float(intervals.mean()) if len(intervals) else np.nan,
        "std_interval_days": float(intervals.std()) if len(intervals) > 1 else 0.0,
        "product_diversity": int(g["stock_code"].nunique()),
        "country_changes": int(g["country"].nunique()),
    })


def _windowed(g: pd.DataFrame, ref: pd.Timestamp, days: int) -> pd.Series:
    """Aggregates over the last `days` before ref."""
    cutoff = ref - pd.Timedelta(days=days)
    recent = g[g["invoice_date"] >= cutoff]
    return pd.Series({
        f"orders_last_{days}d": recent["invoice"].nunique(),
        f"revenue_last_{days}d": float(recent["revenue"].sum()),
        f"items_last_{days}d": int(recent["quantity"].sum()),
    })


def build_customer_features(
    df: pd.DataFrame,
    feature_end: pd.Timestamp,
    country_dummies: list[str] | None = None,
) -> pd.DataFrame:
    """Build features for every customer with at least one purchase before feature_end.

    Parameters
    ----------
    df : cleaned transactions (output of src.data.loader.clean).
    feature_end : reference date — features computed using transactions
        with invoice_date < feature_end.
    country_dummies : optional list of countries to one-hot encode. If None,
        the customer's last-seen country is encoded as a single categorical.

    Returns
    -------
    DataFrame indexed by customer_id with feature columns.
    """
    pre = df[df["invoice_date"] < feature_end]
    grouped = pre.groupby("customer_id")

    rfm = grouped.apply(_basic_rfm, ref=feature_end, include_groups=False)
    beh = grouped.apply(_behavioral, include_groups=False)
    w7 = grouped.apply(_windowed, ref=feature_end, days=7, include_groups=False)
    w30 = grouped.apply(_windowed, ref=feature_end, days=30, include_groups=False)
    w90 = grouped.apply(_windowed, ref=feature_end, days=90, include_groups=False)

    out = pd.concat([rfm, beh, w7, w30, w90], axis=1)

    # ratios — useful nonlinear signals for GBM
    out["revenue_per_order"] = out["monetary"] / out["frequency"].clip(lower=1)
    out["items_per_order"] = out["n_items"] / out["frequency"].clip(lower=1)
    out["recency_over_tenure"] = out["recency_days"] / out["tenure_days"].clip(lower=1)
    out["recent_share_30d"] = out["revenue_last_30d"] / out["monetary"].clip(lower=1e-6)

    last_country = grouped["country"].agg(lambda s: s.iloc[-1])
    if country_dummies:
        for c in country_dummies:
            out[f"country_{c}"] = (last_country == c).astype("int8")
    else:
        out["country"] = last_country.astype("category")

    return out.reset_index()

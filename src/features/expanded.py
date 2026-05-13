"""Expanded Kaggle-style feature engineering.

Ten families, computed from transactions <= feature_end:
  1. multi-horizon aggregates  (7/14/30/60/90/180d + all-time)
  2. time-decay-weighted RFM
  3. inter-purchase interval statistics
  4. cohort
  5. target encoding (handled in target_encoding.py — applied at the comparison
     notebook level so the train/val/test split is honored)
  6. item-side rollups joined to customer
  7. basket statistics (entropy, gini, max)
  8. sequence features (basket-to-basket Jaccard similarity)
  9. behavioral ratios
 10. time-of-day / day-of-week shares

The baseline `build_customer_features` in build_features.py is still callable for
A/B comparison — this module produces a superset.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# -------- 1. Multi-horizon aggregates --------

HORIZONS = (7, 14, 30, 60, 90, 180)


def _windowed_aggs(g: pd.DataFrame, ref: pd.Timestamp, horizons: Iterable[int]) -> pd.Series:
    out = {}
    for d in horizons:
        recent = g[g["invoice_date"] >= ref - pd.Timedelta(days=d)]
        out[f"orders_last_{d}d"] = recent["invoice"].nunique()
        out[f"revenue_last_{d}d"] = float(recent["revenue"].sum())
        out[f"items_last_{d}d"] = int(recent["quantity"].sum())
        out[f"unique_skus_last_{d}d"] = recent["stock_code"].nunique()
    return pd.Series(out)


# -------- 2. Time-decay weighted RFM --------


def _decay_features(g: pd.DataFrame, ref: pd.Timestamp, half_life_days: float = 30) -> pd.Series:
    """Exponential decay weights more recent activity higher."""
    lam = np.log(2) / half_life_days
    age_days = (ref - g["invoice_date"]).dt.total_seconds() / 86400
    w = np.exp(-lam * age_days)
    return pd.Series({
        "decay_revenue": float((w * g["revenue"]).sum()),
        "decay_items": float((w * g["quantity"]).sum()),
        "decay_n_events": float(w.sum()),
    })


# -------- 3. Inter-purchase interval stats --------


def _interval_stats(g: pd.DataFrame) -> pd.Series:
    orders = g.groupby("invoice")["invoice_date"].min().sort_values()
    if len(orders) < 2:
        return pd.Series({
            "interval_mean": np.nan, "interval_std": 0.0, "interval_min": np.nan,
            "interval_max": np.nan, "last_gap_over_mean": np.nan,
        })
    gaps = orders.diff().dt.days.dropna()
    last_gap = (orders.iloc[-1] - orders.iloc[-2]).days
    mean = float(gaps.mean())
    return pd.Series({
        "interval_mean": mean,
        "interval_std": float(gaps.std() or 0.0),
        "interval_min": float(gaps.min()),
        "interval_max": float(gaps.max()),
        "last_gap_over_mean": last_gap / mean if mean > 0 else np.nan,
    })


# -------- 4. Cohort --------


def _cohort_features(g: pd.DataFrame, ref: pd.Timestamp) -> pd.Series:
    first = g["invoice_date"].min()
    return pd.Series({
        "cohort_month": first.month,
        "cohort_year": first.year,
        "days_since_cohort": (ref - first).days,
    })


# -------- 6. Item-side rollups joined to customer --------


def _item_popularity(pre: pd.DataFrame) -> pd.DataFrame:
    """Per-stock_code popularity computed from pre-feature_end transactions only."""
    pop = pre.groupby("stock_code").agg(
        pop_n_orders=("invoice", "nunique"),
        pop_n_customers=("customer_id", "nunique"),
        pop_avg_price=("price", "mean"),
    )
    pop["pop_rank"] = pop["pop_n_orders"].rank(ascending=False, method="dense")
    pop["pop_log_n_orders"] = np.log1p(pop["pop_n_orders"])
    return pop.reset_index()


def _customer_item_rollups(g: pd.DataFrame, item_features: pd.DataFrame) -> pd.Series:
    cust_items = g[["stock_code"]].merge(item_features, on="stock_code", how="left")
    return pd.Series({
        "avg_item_popularity": float(cust_items["pop_log_n_orders"].mean()),
        "avg_item_price_tier": float(cust_items["pop_avg_price"].mean()),
        "max_item_pop_rank": float(cust_items["pop_rank"].min()),  # min rank = most popular
        "long_tail_share": float((cust_items["pop_rank"] > 1000).mean()),
    })


# -------- 7. Basket statistics --------


def _gini(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    cum = np.cumsum(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * cum[-1]) / (n * cum[-1] + 1e-12))


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _basket_stats(g: pd.DataFrame) -> pd.Series:
    baskets = g.groupby("invoice").agg(
        basket_revenue=("revenue", "sum"),
        basket_items=("quantity", "sum"),
        basket_skus=("stock_code", "nunique"),
    )
    return pd.Series({
        "basket_revenue_max": float(baskets["basket_revenue"].max()),
        "basket_revenue_std": float(baskets["basket_revenue"].std() or 0.0),
        "basket_revenue_gini": _gini(baskets["basket_revenue"].values),
        "basket_size_entropy": _entropy(baskets["basket_items"].values),
        "unique_sku_rate": float(g["stock_code"].nunique() / max(g["quantity"].sum(), 1)),
    })


# -------- 8. Sequence features --------


def _basket_sequence_features(g: pd.DataFrame) -> pd.Series:
    """Avg Jaccard similarity between consecutive baskets — repeat-purchase intensity."""
    baskets = g.groupby("invoice").agg(
        sku_set=("stock_code", lambda s: frozenset(s)),
        order_date=("invoice_date", "min"),
    ).sort_values("order_date")
    sets = list(baskets["sku_set"])
    if len(sets) < 2:
        return pd.Series({"avg_basket_jaccard": np.nan, "repeat_sku_rate": np.nan})
    jaccs = []
    for a, b in zip(sets[:-1], sets[1:], strict=True):
        u = a | b
        jaccs.append(len(a & b) / len(u) if u else 0.0)
    all_skus = [s for fs in sets for s in fs]
    unique = len(set(all_skus))
    return pd.Series({
        "avg_basket_jaccard": float(np.mean(jaccs)),
        "repeat_sku_rate": 1.0 - unique / max(len(all_skus), 1),
    })


# -------- 9. Behavioral ratios --------


def _ratios(rfm: pd.Series, decay: pd.Series, multi: pd.Series) -> pd.Series:
    monetary = rfm.get("monetary", 0.0)
    return pd.Series({
        "ratio_recent30_lifetime_rev": multi["revenue_last_30d"] / (monetary + 1e-6),
        "ratio_recent7_lifetime_rev": multi["revenue_last_7d"] / (monetary + 1e-6),
        "decay_rev_over_lifetime": decay["decay_revenue"] / (monetary + 1e-6),
    })


# -------- 10. Time-of-day / day-of-week --------


def _temporal_shares(g: pd.DataFrame) -> pd.Series:
    dow = g["invoice_date"].dt.dayofweek.value_counts(normalize=True).reindex(range(7), fill_value=0)
    hours = g["invoice_date"].dt.hour
    morning = ((hours >= 6) & (hours < 12)).mean()
    afternoon = ((hours >= 12) & (hours < 18)).mean()
    evening = (hours >= 18).mean()
    return pd.Series({
        **{f"dow_{i}_share": float(dow.loc[i]) for i in range(7)},
        "share_morning": float(morning),
        "share_afternoon": float(afternoon),
        "share_evening": float(evening),
        "weekend_share": float(dow.loc[5] + dow.loc[6]),
    })


# -------- Master assembly --------


def _basic_rfm(g: pd.DataFrame, ref: pd.Timestamp) -> pd.Series:
    last = g["invoice_date"].max()
    first = g["invoice_date"].min()
    return pd.Series({
        "recency_days": (ref - last).days,
        "tenure_days": (ref - first).days,
        "frequency": g["invoice"].nunique(),
        "monetary": float(g["revenue"].sum()),
        "n_items": int(g["quantity"].sum()),
        "product_diversity": int(g["stock_code"].nunique()),
    })


def build_expanded_features(
    df: pd.DataFrame,
    feature_end: pd.Timestamp,
    country_dummies: list[str] | None = None,
    horizons: Iterable[int] = HORIZONS,
    decay_half_life_days: float = 30,
) -> pd.DataFrame:
    """Build expanded feature table indexed by customer_id, using only rows < feature_end."""
    pre = df[df["invoice_date"] < feature_end].copy()
    item_feats = _item_popularity(pre)

    rows = []
    for cust, g in pre.groupby("customer_id", sort=False):
        rfm = _basic_rfm(g, feature_end)
        multi = _windowed_aggs(g, feature_end, horizons)
        decay = _decay_features(g, feature_end, half_life_days=decay_half_life_days)
        intervals = _interval_stats(g)
        cohort = _cohort_features(g, feature_end)
        item_roll = _customer_item_rollups(g, item_feats)
        basket = _basket_stats(g)
        seq = _basket_sequence_features(g)
        ratios = _ratios(rfm, decay, multi)
        temporal = _temporal_shares(g)
        last_country = g["country"].iloc[-1]

        row = pd.concat([rfm, multi, decay, intervals, cohort, item_roll, basket, seq, ratios, temporal])
        row["customer_id"] = int(cust)
        row["last_country"] = last_country
        rows.append(row)

    out = pd.DataFrame(rows)
    if country_dummies:
        for c in country_dummies:
            out[f"country_{c}"] = (out["last_country"] == c).astype("int8")

    front = ["customer_id", "last_country"]
    return out[front + [c for c in out.columns if c not in front]].reset_index(drop=True)

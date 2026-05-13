"""Business-oriented evaluation slices.

Reports performance specifically on the high-churn segment — the population
the retention system is meant to serve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .churn_metrics import evaluate


def evaluate_by_risk_segment(
    customer_id: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    quantiles: list[float] = (0.5, 0.8, 0.95),
) -> pd.DataFrame:
    """Slice metrics by churn-risk decile.

    For each quantile, take customers with p_churn >= quantile and report base rate,
    precision, recall, F1.
    """
    out = []
    for q in quantiles:
        thr = float(np.quantile(y_score, q))
        mask = y_score >= thr
        if mask.sum() == 0:
            continue
        m = evaluate(y_true[mask], y_score[mask], threshold=thr)
        out.append({"quantile": q, "threshold": thr, "n": int(mask.sum()), **m})
    return pd.DataFrame(out)


def retention_impact(
    high_risk_customers: pd.DataFrame,
    actually_churned: set[int],
    recommended_per_customer: dict[int, list[str]],
    purchased_in_label_window: pd.DataFrame,
) -> dict:
    """Simulate how well our top-K hit products the customer actually bought.

    Returns the share of high-risk customers for whom at least one recommended product
    was actually purchased in the label window — a proxy for retention lift.
    """
    purchased_by_cust = purchased_in_label_window.groupby("customer_id")["stock_code"].apply(set).to_dict()
    hit, total = 0, 0
    for cust_id in high_risk_customers["customer_id"]:
        recs = recommended_per_customer.get(int(cust_id), [])
        bought = purchased_by_cust.get(int(cust_id), set())
        if not recs or not bought:
            continue
        total += 1
        if set(recs) & bought:
            hit += 1
    return {
        "high_risk_customers": int(len(high_risk_customers)),
        "with_recs_and_purchases": total,
        "hit_rate": (hit / total) if total else 0.0,
        "churned_in_set": int(len(set(high_risk_customers["customer_id"]) & actually_churned)),
    }

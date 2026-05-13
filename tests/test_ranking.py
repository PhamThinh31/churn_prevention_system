"""Tests for the LightGBM ranker plumbing.

The model itself is a thin wrapper around LGBMRanker; we mostly test the
data-shaping helpers (`build_pairs`, `rerank`) which is where bugs hide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.ranking.lgbm_ranker import build_pairs, rerank, train_ranker


def _toy_inputs():
    candidates = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2],
        "stock_code":  ["A", "B", "C", "A", "D", "E"],
        "rank":        [0, 1, 2, 0, 1, 2],
        "score":       [0.9, 0.5, 0.3, 0.8, 0.6, 0.4],
    })
    label_window = pd.DataFrame({
        "customer_id": [1, 2],
        "stock_code":  ["A", "D"],
    })
    cust_feats = pd.DataFrame({
        "customer_id":  [1, 2],
        "recency_days": [10, 5],
        "frequency":    [3, 4],
        "monetary":     [200.0, 150.0],
    })
    item_feats = pd.DataFrame({
        "stock_code":     ["A", "B", "C", "D", "E"],
        "item_log_orders":[1.0, 0.5, 0.3, 0.8, 0.4],
        "item_avg_price": [5.0, 3.0, 4.0, 6.0, 7.0],
    })
    churn = pd.DataFrame({"customer_id": [1, 2], "p_churn": [0.85, 0.2]})
    return candidates, label_window, cust_feats, item_feats, churn


def test_build_pairs_labels_relevant_correctly():
    cand, lw, cf, itf, ch = _toy_inputs()
    pairs = build_pairs(cand, lw, cf, itf, ch)

    # (1, A) and (2, D) were purchased → relevance 1
    rel = pairs.set_index(["customer_id", "stock_code"])["relevance"]
    assert rel.loc[(1, "A")] == 1
    assert rel.loc[(2, "D")] == 1
    # Everything else is 0
    assert rel.loc[(1, "B")] == 0
    assert rel.loc[(1, "C")] == 0
    assert rel.loc[(2, "A")] == 0
    assert rel.loc[(2, "E")] == 0


def test_build_pairs_joins_features():
    cand, lw, cf, itf, ch = _toy_inputs()
    pairs = build_pairs(cand, lw, cf, itf, ch)
    assert "p_churn" in pairs.columns
    assert "recency_days" in pairs.columns
    assert "item_log_orders" in pairs.columns
    # derived columns
    assert "log_score" in pairs.columns
    assert "inv_rank" in pairs.columns
    # inv_rank monotonically decreases with rank
    sorted_pairs = pairs.sort_values(["customer_id", "rank"])
    assert (sorted_pairs.groupby("customer_id")["inv_rank"].diff().dropna() <= 0).all()


def test_train_and_rerank():
    """Training should succeed, and rerank should return top-k per customer."""
    cand, lw, cf, itf, ch = _toy_inputs()
    pairs = build_pairs(cand, lw, cf, itf, ch)
    feature_cols = ["score", "log_score", "inv_rank", "rank", "p_churn",
                    "recency_days", "frequency", "monetary",
                    "item_log_orders", "item_avg_price"]
    model = train_ranker(pairs, feature_cols)
    out = rerank(model, pairs, feature_cols, k=2)
    assert "rank_score" in out.columns
    assert "final_rank" in out.columns
    # k=2 → each customer keeps at most 2 rows
    assert out.groupby("customer_id").size().max() == 2
    # final_rank starts at 0
    assert (out["final_rank"] >= 0).all()


def test_rerank_orders_by_score_descending():
    """Within each customer, final_rank should follow rank_score descending."""
    cand, lw, cf, itf, ch = _toy_inputs()
    pairs = build_pairs(cand, lw, cf, itf, ch)
    cols = ["score", "log_score", "inv_rank", "rank", "p_churn"]
    model = train_ranker(pairs, cols)
    out = rerank(model, pairs, cols, k=3).sort_values(["customer_id", "final_rank"])
    for cust, g in out.groupby("customer_id"):
        scores = g["rank_score"].values
        assert np.all(np.diff(scores) <= 1e-6), \
            f"Customer {cust}: rank_score not non-increasing: {scores}"

"""Tests for the recsys evaluation metrics.

Recall@K and NDCG@K need correct behavior at the boundary cases (empty inputs,
no overlap, perfect overlap) before we can trust them on real predictions.
"""
from __future__ import annotations

import math

import pandas as pd

from src.evaluation.recsys_metrics import (
    evaluate_recsys,
    ndcg_at_k,
    recall_at_k,
)

# ───────────────────────── recall_at_k ─────────────────────────


def test_recall_perfect_match():
    pred = [["A", "B", "C"]]
    rel = [{"A", "B"}]
    assert recall_at_k(pred, rel, k=3) == 1.0


def test_recall_half_match():
    pred = [["A", "X", "Y", "Z"]]
    rel = [{"A", "B"}]
    assert recall_at_k(pred, rel, k=4) == 0.5


def test_recall_no_match():
    pred = [["X", "Y"]]
    rel = [{"A", "B"}]
    assert recall_at_k(pred, rel, k=2) == 0.0


def test_recall_respects_k():
    """Item at position 5 should not count when k=3."""
    pred = [["X", "Y", "Z", "W", "A"]]
    rel = [{"A"}]
    assert recall_at_k(pred, rel, k=3) == 0.0
    assert recall_at_k(pred, rel, k=5) == 1.0


def test_recall_empty_input():
    assert recall_at_k([], [], k=5) == 0.0


def test_recall_skips_users_with_no_relevant_items():
    """A user with empty `relevant` shouldn't drag the mean down."""
    pred = [["A"], ["A"]]
    rel = [{"A"}, set()]
    # Only the first user counts. Recall@1 = 1/1.
    assert recall_at_k(pred, rel, k=1) == 1.0


def test_recall_averages_across_users():
    pred = [["A", "Q"], ["Y", "X"]]
    rel = [{"A"}, {"X"}]
    # k=2: user 1 hits A (1/1), user 2 hits X (1/1) → mean 1.0
    # k=1: user 1 hits A (1/1), user 2 only sees Y (0/1) → mean 0.5
    assert recall_at_k(pred, rel, k=2) == 1.0
    assert recall_at_k(pred, rel, k=1) == 0.5


# ───────────────────────── ndcg_at_k ─────────────────────────


def test_ndcg_perfect_order():
    pred = [["A", "B"]]
    rel = [{"A", "B"}]
    assert math.isclose(ndcg_at_k(pred, rel, k=2), 1.0, rel_tol=1e-6)


def test_ndcg_reversed_still_relevant():
    """If all relevant items are in top-k, NDCG is 1.0 regardless of order *within*."""
    pred = [["A", "B"]]
    rel = [{"A", "B"}]
    assert math.isclose(ndcg_at_k(pred, rel, k=2), 1.0, rel_tol=1e-6)


def test_ndcg_zero():
    pred = [["X"]]
    rel = [{"A"}]
    assert ndcg_at_k(pred, rel, k=1) == 0.0


def test_ndcg_rewards_higher_position():
    """A hit at position 1 should score higher than the same hit at position 3."""
    pred_top = [["A", "X", "Y"]]
    pred_bottom = [["X", "Y", "A"]]
    rel = [{"A"}]
    score_top = ndcg_at_k(pred_top, rel, k=3)
    score_bottom = ndcg_at_k(pred_bottom, rel, k=3)
    assert score_top > score_bottom


# ───────────────────────── evaluate_recsys ─────────────────────────


def test_evaluate_recsys_end_to_end():
    """End-to-end on dataframes — both metrics at multiple Ks."""
    recs = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2],
        "stock_code": ["A", "B", "C", "D", "E"],
    })
    gt = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "stock_code": ["A", "X", "D"],
    })
    metrics = evaluate_recsys(recs, gt, ks=(1, 3))
    # Customer 1: relevant={A,X}, predicted top-3=A,B,C → recall@1=0.5 (A is in top-1), @3=0.5
    # Customer 2: relevant={D}, predicted top-3=D,E   → recall@1=1.0, @3=1.0
    # Mean: @1 = 0.75, @3 = 0.75
    assert metrics["recall@1"] == 0.75
    assert metrics["recall@3"] == 0.75
    assert "ndcg@1" in metrics and "ndcg@3" in metrics

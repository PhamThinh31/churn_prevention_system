"""Unit tests for the decision layer.

Asserts that the tier/discount mapping follows the documented policy and that
threshold customization works for the business team's tuning needs.
"""
from __future__ import annotations

from src.decision.retention import RetentionAction, decide


def test_high_risk():
    a = decide(123, p_churn=0.85, top_products=["A", "B"])
    assert isinstance(a, RetentionAction)
    assert a.customer_id == 123
    assert a.p_churn == 0.85
    assert a.risk_tier == "high"
    assert a.suggested_discount_pct == 20
    assert a.top_products == ["A", "B"]


def test_medium_risk():
    a = decide(1, 0.5, [])
    assert a.risk_tier == "medium"
    assert a.suggested_discount_pct == 10


def test_low_risk():
    a = decide(1, 0.1, [])
    assert a.risk_tier == "low"
    assert a.suggested_discount_pct == 0


def test_boundary_high_inclusive():
    """p_churn exactly at the high threshold should be classified high."""
    a = decide(1, 0.7, [])
    assert a.risk_tier == "high"


def test_boundary_medium_inclusive():
    a = decide(1, 0.4, [])
    assert a.risk_tier == "medium"


def test_thresholds_customizable():
    """Business team can pass tighter cutoffs without code change."""
    a = decide(1, 0.5, [], high_threshold=0.9, medium_threshold=0.6)
    assert a.risk_tier == "low"  # 0.5 < 0.6

    b = decide(1, 0.95, [], high_threshold=0.9, medium_threshold=0.6)
    assert b.risk_tier == "high"


def test_p_churn_cast_to_float():
    """Accept numpy floats etc. without surprises."""
    import numpy as np
    a = decide(1, np.float64(0.72), [])
    assert isinstance(a.p_churn, float)
    assert a.risk_tier == "high"

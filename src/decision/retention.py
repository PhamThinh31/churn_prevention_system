"""Decision layer — convert churn probability + ranked products into a retention action.

This is the policy layer that sits between the ML predictions and the campaign system.
It maps a per-customer churn probability to one of three risk tiers and attaches a
discount level appropriate to that tier. The product ranking comes from the
retrieval + ranking stack and is passed through unmodified.

Tier thresholds are intentionally simple (two cutoffs) so non-technical stakeholders
can argue about them without touching ML code. The discount percentages are placeholders
for whatever retention budget the business team decides.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetentionAction:
    """A single recommended retention action for one customer.

    Attributes
    ----------
    customer_id : int
        The customer this action targets.
    p_churn : float
        Predicted probability of churn in the next horizon window (default 90 days).
    risk_tier : str
        ``"high"`` | ``"medium"`` | ``"low"``. Derived from ``p_churn`` via :func:`decide`.
    top_products : list[str]
        Stock codes the customer is most likely to buy, in rank order. Comes from the
        retrieval + ranking + (optional) LLM rerank pipeline.
    suggested_discount_pct : int
        Discount percentage attached to the offer. 20 for high, 10 for medium, 0 for low.
    """

    customer_id: int
    p_churn: float
    risk_tier: str
    top_products: list[str]
    suggested_discount_pct: int


def decide(
    customer_id: int,
    p_churn: float,
    top_products: list[str],
    high_threshold: float = 0.7,
    medium_threshold: float = 0.4,
) -> RetentionAction:
    """Map a churn probability + product ranking to a concrete retention action.

    Tiering rules:

    - ``p_churn >= high_threshold``    → ``"high"``   risk, 20% discount
    - ``p_churn >= medium_threshold``  → ``"medium"`` risk, 10% discount
    - otherwise                        → ``"low"``    risk, no action

    The thresholds are exposed as parameters so the business team can tune them
    without touching the modeling code. The dashboard chart
    ``churn_distribution.png`` visualizes how the test population splits at the
    default values.
    """
    if p_churn >= high_threshold:
        tier, discount = "high", 20
    elif p_churn >= medium_threshold:
        tier, discount = "medium", 10
    else:
        tier, discount = "low", 0
    return RetentionAction(
        customer_id=customer_id,
        p_churn=float(p_churn),
        risk_tier=tier,
        top_products=top_products,
        suggested_discount_pct=discount,
    )

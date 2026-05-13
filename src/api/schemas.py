"""Pydantic schemas for the REST API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    customer_id: int
    top_k: int = Field(default=10, ge=1, le=50)
    use_llm_reranker: bool = False


class ProductRec(BaseModel):
    stock_code: str
    score: float
    rank: int


class RecommendResponse(BaseModel):
    customer_id: int
    p_churn: float
    risk_tier: str
    suggested_discount_pct: int
    top_products: list[ProductRec]


class HealthResponse(BaseModel):
    status: str
    components: dict[str, bool]

"""Integration tests for the FastAPI inference service.

Uses FastAPI's TestClient — no live server needed. We verify endpoint shapes,
validation, and the not-loaded path (when artifacts haven't been trained yet).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Import inside the fixture so module-import errors surface as test failures.
    from src.api.app import app
    return TestClient(app)


def test_health_returns_components(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert body["status"] in {"ok", "degraded"}
    assert "components" in body
    # Components should be a bool map of expected artifacts.
    assert set(body["components"]).issubset({"churn_scores", "sasrec", "faiss", "transactions"})


def test_recommend_validates_body(client):
    # Missing required field
    r = client.post("/recommend", json={})
    assert r.status_code == 422  # Pydantic validation error


def test_recommend_rejects_invalid_top_k(client):
    r = client.post("/recommend", json={"customer_id": 1, "top_k": 0})
    assert r.status_code == 422

    r = client.post("/recommend", json={"customer_id": 1, "top_k": 100})
    assert r.status_code == 422


def test_recommend_503_when_not_loaded(client):
    """When pipeline artifacts aren't present, /recommend must return 503,
    not crash or quietly return empty."""
    from src.api.app import state
    if state.ready:
        pytest.skip("Artifacts are loaded — this test only covers the cold-start path.")

    r = client.post("/recommend", json={"customer_id": 12347, "top_k": 5})
    assert r.status_code == 503
    assert "not loaded" in r.text.lower() or "train" in r.text.lower()


def test_recommend_404_for_unknown_customer(client):
    """When artifacts ARE loaded but the customer is unknown, /recommend should
    return 404. Skipped if artifacts aren't loaded."""
    from src.api.app import state
    if not state.ready:
        pytest.skip("Artifacts not loaded — covered by test_recommend_503_when_not_loaded.")
    r = client.post("/recommend", json={"customer_id": -99999999, "top_k": 5})
    assert r.status_code == 404


def test_recommend_happy_path_when_loaded(client):
    """If artifacts are loaded, a known customer_id should get a valid response."""
    from src.api.app import state
    if not state.ready or state.churn_scores is None or len(state.churn_scores) == 0:
        pytest.skip("Artifacts not loaded.")
    cid = int(state.churn_scores.index[0])
    r = client.post("/recommend", json={"customer_id": cid, "top_k": 5, "use_llm_reranker": False})
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == cid
    assert body["risk_tier"] in {"high", "medium", "low"}
    assert 0 <= body["suggested_discount_pct"] <= 100
    assert len(body["top_products"]) <= 5
    for p in body["top_products"]:
        assert {"stock_code", "score", "rank"} <= set(p)

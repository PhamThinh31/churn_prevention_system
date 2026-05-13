"""Integration test on a small slice of the real dataset.

Validates that loader → clean → splits → labels run end-to-end and produce
non-empty, sane outputs. Skipped if the raw CSV is missing (e.g. in CI).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.labels import make_churn_labels
from src.data.loader import clean, load_raw
from src.data.splits import build_windows

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "online_retail_II.csv"


@pytest.mark.skipif(not RAW.exists(), reason="raw CSV not present")
def test_loader_clean_roundtrip():
    raw = load_raw(RAW)
    assert len(raw) > 100
    assert {"invoice", "stock_code", "customer_id", "invoice_date", "quantity", "price"} <= set(raw.columns)

    df = clean(raw)
    assert (df["quantity"] > 0).all()
    assert (df["price"] > 0).all()
    assert df["customer_id"].notna().all()
    assert df["customer_id"].dtype == "int64"


@pytest.mark.skipif(not RAW.exists(), reason="raw CSV not present")
def test_windows_and_labels():
    df = clean(load_raw(RAW))
    windows = build_windows(df, horizon_days=90, n_windows=3)
    assert len(windows) == 3
    assert windows[0].feature_end < windows[1].feature_end < windows[2].feature_end

    labels = make_churn_labels(df, windows[0])
    assert {"customer_id", "churn", "window"} <= set(labels.columns)
    assert labels["churn"].isin([0, 1]).all()
    assert 0.0 < labels["churn"].mean() < 1.0

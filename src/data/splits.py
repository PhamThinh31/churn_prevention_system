"""Time-based train/val/test splits for churn labeling.

Each window is (feature_end, label_end). Features are computed on all
transactions up to feature_end. Label is 1 if the customer made no purchase
between feature_end and label_end (= feature_end + horizon_days).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class Window:
    name: str
    feature_end: pd.Timestamp
    label_end: pd.Timestamp


def build_windows(df: pd.DataFrame, horizon_days: int = 90, n_windows: int = 3) -> list[Window]:
    """Roll back from the end of the dataset by `horizon_days` per window.

    The most recent window (test) ends at the last transaction date.
    Earlier windows (val, train) are offset by one horizon each.
    """
    max_date = df["invoice_date"].max().normalize()
    names = ["train", "val", "test"][:n_windows]
    windows = []
    for i, name in enumerate(reversed(names)):  # build from latest backwards
        label_end = max_date - timedelta(days=horizon_days * i)
        feature_end = label_end - timedelta(days=horizon_days)
        windows.append(Window(name=name, feature_end=feature_end, label_end=label_end))
    return list(reversed(windows))


def customers_active_before(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Index:
    """Customers with at least one transaction strictly before cutoff."""
    pre = df[df["invoice_date"] < cutoff]
    return pre["customer_id"].drop_duplicates().sort_values().reset_index(drop=True)

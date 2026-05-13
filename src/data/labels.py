"""Construct churn labels per the spec: churn = no purchase in next N days."""
from __future__ import annotations

import pandas as pd

from .splits import Window


def make_churn_labels(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Return one row per customer active before `window.feature_end`.

    Columns:
      customer_id, last_purchase_pre, churn (1 = no purchase in label window)
    """
    pre = df[df["invoice_date"] < window.feature_end]
    post = df[(df["invoice_date"] >= window.feature_end) & (df["invoice_date"] < window.label_end)]

    last_pre = pre.groupby("customer_id")["invoice_date"].max().rename("last_purchase_pre")
    returned = post["customer_id"].drop_duplicates()

    labels = last_pre.reset_index()
    labels["churn"] = (~labels["customer_id"].isin(returned)).astype("int8")
    labels["window"] = window.name
    return labels

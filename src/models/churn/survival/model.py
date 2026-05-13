"""Survival-analysis churn model using lifelines CoxPH.

Event = "stopped purchasing" (no transaction in `horizon_days` after a reference date).
Duration = days from first purchase to last observed purchase (or to censoring at feature_end).

We fit CoxPH on the training window's customer features, predict the survival
probability at `horizon_days`, and report churn_score = 1 - S(horizon_days).
"""
from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter


def build_survival_frame(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Combine features + labels into a survival-ready frame.

    Duration is `tenure_days - recency_days` (active span), with a minimum of 1
    to satisfy CoxPH's positivity requirement.
    Event = 1 if churn = 1, else 0 (censored — they're still active).
    """
    df = features.merge(labels[["customer_id", "churn"]], on="customer_id", how="inner")
    df["duration"] = (df["tenure_days"] - df["recency_days"]).clip(lower=1)
    df["event"] = df["churn"].astype("int8")

    keep = ["duration", "event"] + [c for c in feature_cols if c in df.columns]
    return df[keep].dropna()


class CoxChurn:
    def __init__(self, penalizer: float = 0.01):
        self.cph = CoxPHFitter(penalizer=penalizer)
        self.feature_cols: list[str] | None = None

    def fit(self, frame: pd.DataFrame) -> CoxChurn:
        self.feature_cols = [c for c in frame.columns if c not in {"duration", "event"}]
        self.cph.fit(frame, duration_col="duration", event_col="event", show_progress=False)
        return self

    def churn_score(self, features: pd.DataFrame, horizon_days: int = 90) -> pd.Series:
        assert self.feature_cols is not None, "Call fit() first."
        X = features[self.feature_cols].dropna()
        sf = self.cph.predict_survival_function(X, times=[horizon_days])
        # sf has horizons as index, customers as columns
        s = sf.iloc[0].values
        return pd.Series(1 - s, index=X.index, name="churn_score_cox")

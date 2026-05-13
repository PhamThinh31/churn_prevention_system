"""BG-NBD churn modeling.

BG-NBD models customer behavior as Poisson purchases with a geometric dropout.
For each customer we compute (frequency, recency, T) summaries up to feature_end,
fit the model on the training window, and convert p-alive at horizon into a
"will-not-purchase-in-next-N-days" probability that we report as churn score.
"""
from __future__ import annotations

import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data


def make_rft_summary(df: pd.DataFrame, feature_end: pd.Timestamp) -> pd.DataFrame:
    """frequency/recency/T summary (lifetimes format) up to feature_end."""
    pre = df[df["invoice_date"] < feature_end].copy()
    return summary_data_from_transaction_data(
        pre,
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        monetary_value_col="revenue",
        observation_period_end=feature_end,
        freq="D",
    )


class BGNBDChurn:
    def __init__(self, penalizer: float = 0.001):
        self.fitter = BetaGeoFitter(penalizer_coef=penalizer)

    def fit(self, summary: pd.DataFrame) -> BGNBDChurn:
        self.fitter.fit(summary["frequency"], summary["recency"], summary["T"])
        return self

    def churn_score(self, summary: pd.DataFrame, horizon_days: int = 90) -> pd.Series:
        """Return P(no purchases in next horizon_days) per customer.

        BG-NBD gives expected purchases; we compute 1 - p_alive * P(>=1 purchase | alive).
        A pragmatic approximation: churn_score = 1 - conditional_expected_purchases_clipped.
        """
        f, r, T = summary["frequency"], summary["recency"], summary["T"]
        expected = self.fitter.conditional_expected_number_of_purchases_up_to_time(
            horizon_days, f, r, T
        )
        # Map expected purchases into [0,1] via 1 - exp(-expected) (Poisson P(>=1)),
        # then churn = 1 - P(>=1).
        import numpy as np
        p_return = 1 - np.exp(-expected.clip(lower=0))
        return pd.Series(1 - p_return, index=summary.index, name="churn_score_bgnbd")

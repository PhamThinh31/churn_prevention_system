"""Churn-aware ranking via LightGBM LambdaRank.

Inputs are (customer, candidate_item) pairs with features:
  - retrieval_score (cosine sim from SASRec/FAISS)
  - churn_prob (from chosen churn model)
  - customer-side: recency, frequency, monetary, etc.
  - item-side: popularity, recent_sales, avg_price
  - interaction: rank in retrieval, log(score)

Label: 1 if the customer actually purchased that item in the label window, else 0.
This is implicit-feedback ranking — we treat purchased items as relevant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


def build_pairs(
    candidates: pd.DataFrame,
    interactions_in_label_window: pd.DataFrame,
    customer_features: pd.DataFrame,
    item_features: pd.DataFrame,
    churn_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble training rows from retrieval candidates + label-window purchases.

    candidates: customer_id, stock_code, rank, score (from FAISS)
    interactions_in_label_window: customer_id, stock_code purchases used as positive labels
    customer_features: indexed by customer_id
    item_features: indexed by stock_code (e.g., popularity, recent_sales, avg_price)
    churn_scores: customer_id, p_churn
    """
    pos = interactions_in_label_window[["customer_id", "stock_code"]].drop_duplicates()
    pos["relevance"] = 1

    df = candidates.merge(pos, on=["customer_id", "stock_code"], how="left")
    df["relevance"] = df["relevance"].fillna(0).astype("int8")

    df = df.merge(customer_features, on="customer_id", how="left")
    df = df.merge(item_features, on="stock_code", how="left")
    df = df.merge(churn_scores, on="customer_id", how="left")
    df["log_score"] = np.log1p(df["score"].clip(lower=0))
    df["inv_rank"] = 1.0 / (df["rank"] + 1)
    return df


def train_ranker(train_df: pd.DataFrame, feature_cols: list[str]) -> LGBMRanker:
    """Train a LightGBM LambdaRank model on grouped (customer → candidates) pairs.

    LightGBM expects the rows sorted such that each group's rows are contiguous,
    plus a ``group`` array giving the count per group — we sort by ``customer_id``
    and emit the per-customer size array.

    Parameters
    ----------
    train_df : DataFrame produced by :func:`build_pairs` — must contain ``customer_id``,
        ``relevance``, and every column listed in ``feature_cols``.
    feature_cols : list of column names to use as features.

    Returns
    -------
    Fitted LGBMRanker. Use :func:`rerank` to score new candidates.
    """
    train_df = train_df.sort_values("customer_id").reset_index(drop=True)
    group = train_df.groupby("customer_id").size().values
    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_df[feature_cols].fillna(0), train_df["relevance"], group=group)
    return model


def rerank(
    model: LGBMRanker,
    candidates_with_features: pd.DataFrame,
    feature_cols: list[str],
    k: int = 10,
) -> pd.DataFrame:
    """Score candidates with the trained ranker and keep the top-k per customer.

    The output adds two columns: ``rank_score`` (raw model output, higher = more
    relevant) and ``final_rank`` (0-indexed position within the customer).
    """
    df = candidates_with_features.copy()
    df["rank_score"] = model.predict(df[feature_cols].fillna(0))
    df = df.sort_values(["customer_id", "rank_score"], ascending=[True, False])
    df["final_rank"] = df.groupby("customer_id").cumcount()
    return df[df["final_rank"] < k]

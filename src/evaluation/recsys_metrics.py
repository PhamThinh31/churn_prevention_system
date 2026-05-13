"""Recsys evaluation: Recall@K and NDCG@K with grouped relevance."""
from __future__ import annotations

import numpy as np
import pandas as pd


def recall_at_k(predicted: list[list], relevant: list[set], k: int) -> float:
    if not predicted:
        return 0.0
    hits = []
    for p, r in zip(predicted, relevant, strict=False):
        if not r:
            continue
        hits.append(len(set(p[:k]) & r) / len(r))
    return float(np.mean(hits)) if hits else 0.0


def ndcg_at_k(predicted: list[list], relevant: list[set], k: int) -> float:
    """Binary-relevance NDCG@K."""
    if not predicted:
        return 0.0
    scores = []
    for p, r in zip(predicted, relevant, strict=False):
        if not r:
            continue
        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(p[:k]) if item in r)
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(r))))
        scores.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_recsys(recs_df: pd.DataFrame, ground_truth_df: pd.DataFrame, ks: list[int] = (5, 10, 20)) -> dict:
    """recs_df: customer_id, stock_code (in rank order). ground_truth_df: customer_id, stock_code."""
    gt = ground_truth_df.groupby("customer_id")["stock_code"].apply(set).to_dict()
    rec = recs_df.groupby("customer_id")["stock_code"].apply(list).to_dict()
    customers = list(set(gt) & set(rec))
    pred = [rec[c] for c in customers]
    rel = [gt[c] for c in customers]
    return {
        **{f"recall@{k}": recall_at_k(pred, rel, k) for k in ks},
        **{f"ndcg@{k}": ndcg_at_k(pred, rel, k) for k in ks},
    }

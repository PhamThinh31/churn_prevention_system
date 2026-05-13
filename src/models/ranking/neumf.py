"""NeuMF (Neural Matrix Factorization) ranker — He et al. 2017.

Architecture: GMF (element-wise product of embeddings) + MLP branch, concatenated
and projected to a relevance logit. Trained with BCE on (positive, negative) pairs
sampled from interactions in the label window.

This module satisfies the spec's section 4.5 requirement that the ranking model be
"one of: BPR / NeuMF / Wide & Deep / DeepFM / Sequential models". LGBMRanker
remains available as a strong tabular alternative in src/models/ranking/lgbm_ranker.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class NeuMFConfig:
    emb_dim: int = 32
    mlp_dims: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 4096
    neg_ratio: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NeuMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, config: NeuMFConfig):
        super().__init__()
        d = config.emb_dim
        self.user_gmf = nn.Embedding(n_users, d)
        self.item_gmf = nn.Embedding(n_items, d)
        self.user_mlp = nn.Embedding(n_users, d)
        self.item_mlp = nn.Embedding(n_items, d)

        layers, in_dim = [], d * 2
        for h in config.mlp_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d + config.mlp_dims[-1], 1)

        for emb in (self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        gmf = self.user_gmf(u) * self.item_gmf(i)
        mlp = self.mlp(torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=-1))
        return self.head(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


class _Pairs(Dataset):
    def __init__(self, u: np.ndarray, i: np.ndarray, y: np.ndarray):
        self.u, self.i, self.y = u.astype("int64"), i.astype("int64"), y.astype("float32")

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.y[idx]


def _negative_samples(pos: pd.DataFrame, n_items: int, ratio: int, rng: np.random.Generator) -> pd.DataFrame:
    """For each (user, +item) row, generate `ratio` random negatives the user hasn't bought."""
    by_user = pos.groupby("u")["i"].apply(set).to_dict()
    neg_rows = []
    for row in pos.itertuples():
        seen = by_user[row.u]
        drawn = 0
        while drawn < ratio:
            cand = int(rng.integers(0, n_items))
            if cand not in seen:
                neg_rows.append((row.u, cand, 0.0))
                drawn += 1
    return pd.DataFrame(neg_rows, columns=["u", "i", "y"])


def train_neumf(
    positives: pd.DataFrame,    # cols: customer_id, stock_code
    config: NeuMFConfig | None = None,
    seed: int = 42,
) -> tuple[NeuMF, dict, dict]:
    """Train NeuMF with BCE on positives + sampled negatives.

    Returns (model, user2id, item2id).
    """
    config = config or NeuMFConfig()
    user2id = {u: k for k, u in enumerate(positives["customer_id"].unique())}
    item2id = {it: k for k, it in enumerate(positives["stock_code"].unique())}

    pos = positives.assign(
        u=positives["customer_id"].map(user2id),
        i=positives["stock_code"].map(item2id),
        y=1.0,
    )[["u", "i", "y"]]

    rng = np.random.default_rng(seed)
    neg = _negative_samples(pos, n_items=len(item2id), ratio=config.neg_ratio, rng=rng)
    df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    ds = _Pairs(df["u"].values, df["i"].values, df["y"].values)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    model = NeuMF(len(user2id), len(item2id), config).to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        total, n = 0.0, 0
        model.train()
        for u, i, y in dl:
            u, i, y = u.to(config.device), i.to(config.device), y.to(config.device)
            logit = model(u, i)
            loss = loss_fn(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"neumf epoch {epoch:>2}  loss={total/max(n,1):.4f}")

    return model, user2id, item2id


def score(
    model: NeuMF,
    candidates: pd.DataFrame,   # cols: customer_id, stock_code
    user2id: dict,
    item2id: dict,
    device: str | None = None,
) -> np.ndarray:
    device = device or next(model.parameters()).device
    u = candidates["customer_id"].map(user2id).fillna(0).astype("int64").values
    i = candidates["stock_code"].map(item2id).fillna(0).astype("int64").values
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(u, device=device),
            torch.tensor(i, device=device),
        ).cpu().numpy()
    return logits

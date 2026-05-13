"""Train SASRec on customer purchase sequences."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import CausalNextDataset, Vocab
from .sasrec import SASRec


@dataclass
class TrainConfig:
    emb_dim: int = 64
    n_heads: int = 2
    n_layers: int = 2
    max_len: int = 50
    dropout: float = 0.2
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def encode_sessions(raw_sessions: list[list], vocab: Vocab) -> list[list[int]]:
    return [[vocab.item2id[i] for i in s if i in vocab.item2id] for s in raw_sessions]


def train_sasrec(
    raw_sessions: list[list],
    config: TrainConfig | None = None,
) -> tuple[SASRec, Vocab]:
    config = config or TrainConfig()

    vocab = Vocab.from_sessions(raw_sessions)
    enc = encode_sessions(raw_sessions, vocab)
    ds = CausalNextDataset(enc, max_len=config.max_len, pad_id=vocab.pad_id)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=0)

    model = SASRec(
        vocab_size=vocab.vocab_size,
        emb_dim=config.emb_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_len=config.max_len,
        dropout=config.dropout,
        pad_id=vocab.pad_id,
    ).to(config.device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    for epoch in range(config.epochs):
        model.train()
        total, n_batches = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(config.device), yb.to(config.device)
            h = model(xb)
            logits = model.logits(h)
            loss = loss_fn(logits.view(-1, vocab.vocab_size), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        print(f"epoch {epoch:>3}  loss={total/max(n_batches,1):.4f}")

    return model, vocab


def save(model: SASRec, vocab: Vocab, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "sasrec.pt")
    torch.save({"item2id": vocab.item2id, "id2item": vocab.id2item,
                "pad_id": vocab.pad_id, "mask_id": vocab.mask_id}, path / "vocab.pt")

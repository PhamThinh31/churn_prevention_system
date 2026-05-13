"""Build per-customer purchase sequences from transactions.

A "sequence" is the ordered list of stock_codes a customer purchased, sorted
by invoice_date (and within an invoice, by description for stable order).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Vocab:
    item2id: dict
    id2item: list
    pad_id: int = 0
    mask_id: int = -1   # set after construction

    @property
    def vocab_size(self) -> int:
        return len(self.id2item) + 2  # +PAD, +MASK

    @classmethod
    def from_sessions(cls, sessions: list[list]) -> Vocab:
        items = sorted({i for s in sessions for i in s})
        id2item = [None] + items  # 0 is PAD
        item2id = {it: i + 1 for i, it in enumerate(items)}
        v = cls(item2id=item2id, id2item=id2item)
        v.mask_id = len(id2item)  # next available id
        return v


def build_sessions(
    df: pd.DataFrame, feature_end: pd.Timestamp, min_len: int = 3
) -> list[tuple[int, list]]:
    """Return [(customer_id, [stock_codes_in_order]), ...] with len >= min_len."""
    pre = df[df["invoice_date"] < feature_end].sort_values(["customer_id", "invoice_date"])
    sessions = []
    for cust, g in pre.groupby("customer_id", sort=False):
        items = g["stock_code"].tolist()
        if len(items) >= min_len:
            sessions.append((int(cust), items))
    return sessions


class MaskedLastDataset(Dataset):
    """For each session, mask the last item and predict it.

    Mirrors the BERT4Rec/SASRec eval setup from the teacher's notebook.
    """

    def __init__(self, sessions: list[list[int]], max_len: int, pad_id: int, mask_id: int):
        self.max_len = max_len
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.sessions = [self._encode(s) for s in sessions]

    def _encode(self, seq: list[int]) -> tuple[torch.Tensor, int]:
        seq = list(seq[-self.max_len:])
        target = seq[-1]
        seq = seq[:-1] + [self.mask_id]
        if len(seq) < self.max_len:
            seq = [self.pad_id] * (self.max_len - len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long), target

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, i):
        x, y = self.sessions[i]
        return x, torch.tensor(y, dtype=torch.long)


def pad_right(seq: list[int], max_len: int, pad_id: int) -> list[int]:
    """Right-pad (or truncate from the left) to exactly max_len.

    Must be right-padding to keep the SASRec encoder NaN-free: with left-padding
    plus a causal mask plus a key-padding mask, position 0 has no valid keys
    and softmax produces NaN, which propagates to every output position.
    """
    seq = list(seq[-max_len:])
    return seq + [pad_id] * (max_len - len(seq))


class CausalNextDataset(Dataset):
    """Causal-LM style: predict next item at each position. Used for SASRec training."""

    def __init__(self, sessions: list[list[int]], max_len: int, pad_id: int):
        self.max_len = max_len
        self.pad_id = pad_id
        self.sessions = []
        for s in sessions:
            s = s[-(max_len + 1):]
            if len(s) < 2:
                continue
            inp = s[:-1]
            tgt = s[1:]
            pad_n = max_len - len(inp)
            if pad_n > 0:
                inp = inp + [pad_id] * pad_n
                tgt = tgt + [pad_id] * pad_n
            self.sessions.append((torch.tensor(inp, dtype=torch.long),
                                  torch.tensor(tgt, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, i):
        return self.sessions[i]

"""Unit tests for the SASRec retrieval model and its dataset utilities.

Covers the bugs that bit us:
- Left-padding + causal + key-padding mask used to produce NaN at every output;
  these tests ensure right-padding stays NaN-free.
- All-PAD inputs (a rare edge case) must zero out instead of NaN-poisoning the batch.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.models.retrieval.dataset import (
    CausalNextDataset,
    Vocab,
    pad_right,
)
from src.models.retrieval.sasrec import SASRec

# ───────────────────────── pad_right ─────────────────────────


def test_pad_right_shorter_than_max():
    out = pad_right([1, 2, 3], max_len=5, pad_id=0)
    assert out == [1, 2, 3, 0, 0]


def test_pad_right_truncates_from_left():
    # Long sequence: keep the most recent max_len items.
    out = pad_right([1, 2, 3, 4, 5, 6, 7], max_len=4, pad_id=0)
    assert out == [4, 5, 6, 7]


def test_pad_right_empty_sequence():
    assert pad_right([], max_len=3, pad_id=0) == [0, 0, 0]


# ───────────────────────── Vocab ─────────────────────────


def test_vocab_assigns_pad_at_zero():
    v = Vocab.from_sessions([["A", "B"], ["B", "C"]])
    assert v.pad_id == 0
    assert v.id2item[0] is None
    assert set(v.id2item[1:]) == {"A", "B", "C"}
    assert v.item2id["A"] >= 1
    assert v.vocab_size == len(v.id2item) + 2   # +PAD already at 0, +MASK


# ───────────────────────── SASRec forward ─────────────────────────


def test_forward_no_pad_no_nan():
    torch.manual_seed(0)
    m = SASRec(vocab_size=50, emb_dim=16, max_len=10, pad_id=0)
    m.eval()
    x = torch.randint(1, 50, (3, 10))   # no PAD anywhere
    h = m(x)
    assert h.shape == (3, 10, 16)
    assert not torch.isnan(h).any()


def test_forward_right_pad_no_nan():
    """The bug fixed in commit 'switch to right-padding': right-padded sequences
    must produce finite outputs at every position."""
    torch.manual_seed(0)
    m = SASRec(vocab_size=50, emb_dim=16, max_len=10, pad_id=0)
    m.eval()
    # Three real items, then PAD
    x = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
    h = m(x)
    assert not torch.isnan(h).any()


# ───────────────────────── user_vector ─────────────────────────


def test_user_vector_shape():
    torch.manual_seed(0)
    m = SASRec(vocab_size=50, emb_dim=16, max_len=10, pad_id=0)
    m.eval()
    x = torch.randint(1, 50, (4, 10))
    u = m.user_vector(x)
    assert u.shape == (4, 16)
    assert not torch.isnan(u).any()


def test_user_vector_picks_last_non_pad_position():
    """For right-padded input, user_vector should select the hidden state at
    lengths-1, not the last position (which is PAD)."""
    torch.manual_seed(0)
    m = SASRec(vocab_size=50, emb_dim=16, max_len=10, pad_id=0)
    m.eval()

    # Same items, different padding pattern → embeddings of position 2 should match
    x_short = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
    h_full = m(x_short)
    u = m.user_vector(x_short)
    assert torch.allclose(u, h_full[:, 2], atol=1e-6), \
        "user_vector should be the hidden state at the last non-PAD position"


def test_user_vector_zeros_all_pad_rows():
    """An entirely-PAD row used to make the whole batch NaN. It should now be
    a zero vector instead."""
    torch.manual_seed(0)
    m = SASRec(vocab_size=50, emb_dim=16, max_len=10, pad_id=0)
    m.eval()
    x = torch.zeros((2, 10), dtype=torch.long)   # all-PAD batch
    u = m.user_vector(x)
    assert torch.equal(u, torch.zeros_like(u))


def test_user_vector_mixed_batch_safe():
    """A mix of normal and all-PAD rows: the normal rows must stay finite."""
    torch.manual_seed(0)
    m = SASRec(vocab_size=50, emb_dim=16, max_len=10, pad_id=0)
    m.eval()
    x = torch.tensor([
        [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # all-PAD — used to poison the batch
        [4, 5, 6, 7, 8, 0, 0, 0, 0, 0],
    ])
    u = m.user_vector(x)
    assert not torch.isnan(u).any()
    assert torch.equal(u[1], torch.zeros(16))


# ───────────────────────── item_matrix ─────────────────────────


def test_item_matrix_shape_matches_vocab():
    m = SASRec(vocab_size=42, emb_dim=8, max_len=5, pad_id=0)
    mat = m.item_matrix()
    assert mat.shape == (42, 8)


# ───────────────────────── CausalNextDataset ─────────────────────────


def test_causal_next_dataset_rightpad():
    """Training inputs must be right-padded so the model never sees left-pad."""
    ds = CausalNextDataset([[1, 2, 3, 4]], max_len=6, pad_id=0)
    assert len(ds) == 1
    inp, tgt = ds[0]
    # input is s[:-1] = [1,2,3], padded to 6 → [1,2,3,0,0,0]
    assert inp.tolist() == [1, 2, 3, 0, 0, 0]
    assert tgt.tolist() == [2, 3, 4, 0, 0, 0]


def test_causal_next_dataset_skips_too_short():
    ds = CausalNextDataset([[7]], max_len=10, pad_id=0)
    assert len(ds) == 0


def test_train_loader_runs_forward():
    """End-to-end smoke: build dataset, run one batch through the model."""
    torch.manual_seed(0)
    sessions = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10]]
    ds = CausalNextDataset(sessions, max_len=5, pad_id=0)
    dl = DataLoader(ds, batch_size=4)
    m = SASRec(vocab_size=20, emb_dim=8, max_len=5, pad_id=0)
    m.eval()
    for inp, _tgt in dl:
        h = m(inp)
        assert h.shape == (inp.size(0), 5, 8)
        assert not torch.isnan(h).any()
        break

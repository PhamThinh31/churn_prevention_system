"""Tests for the FAISS retrieval layer.

Verifies that the index build + search roundtrip is correct, that L2 normalization
handles zero-vectors gracefully, and that the file save/load cycle preserves
search results.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.faiss.index import (
    build_index,
    l2_normalize,
    load_index,
    save_index,
    topk,
)


def test_l2_normalize_unit_norm():
    x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]], dtype="float32")
    out = l2_normalize(x)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_l2_normalize_zero_vector_safe():
    """Zero rows must not produce NaN (clip prevents div-by-zero)."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype="float32")
    out = l2_normalize(x)
    assert not np.isnan(out).any()
    assert (out[0] == 0).all()


def test_build_and_topk_finds_self():
    """Each item should be its own nearest neighbor."""
    rng = np.random.default_rng(0)
    items = l2_normalize(rng.standard_normal((50, 8)).astype("float32"))
    index = build_index(items)
    assert index.ntotal == 50

    scores, ids = topk(index, items[:5], k=3)
    assert ids.shape == (5, 3)
    # The closest match for item i should be item i itself.
    assert np.array_equal(ids[:, 0], np.arange(5))
    assert (scores[:, 0] > 0.99).all()   # cosine with self ≈ 1


def test_topk_respects_k():
    rng = np.random.default_rng(1)
    items = l2_normalize(rng.standard_normal((20, 4)).astype("float32"))
    index = build_index(items)
    queries = items[:3]

    for k in [1, 5, 10, 20]:
        _, ids = topk(index, queries, k=k)
        assert ids.shape == (3, k)
        assert (ids >= 0).all() and (ids < 20).all()


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    items = l2_normalize(rng.standard_normal((30, 6)).astype("float32"))
    index = build_index(items)

    out = tmp_path / "idx.faiss"
    save_index(index, out)
    assert out.exists()

    loaded = load_index(out)
    assert loaded.ntotal == 30

    queries = items[:4]
    s1, i1 = topk(index, queries, k=5)
    s2, i2 = topk(loaded, queries, k=5)
    assert np.array_equal(i1, i2)
    assert np.allclose(s1, s2, atol=1e-6)


@pytest.mark.parametrize("dim", [8, 32, 64])
def test_dimensions(dim):
    """Index should accept any reasonable embedding dim."""
    rng = np.random.default_rng(dim)
    items = l2_normalize(rng.standard_normal((100, dim)).astype("float32"))
    index = build_index(items)
    _, ids = topk(index, items[:1], k=5)
    assert ids.shape == (1, 5)

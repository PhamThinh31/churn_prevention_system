"""Build / load a FAISS index over item embeddings, retrieve top-K per user vector.

Used by the retrieval stage of the pipeline: SASRec trains item embeddings, this
module builds an inner-product index over the L2-normalized embeddings (so
inner product = cosine similarity), and the API queries it with the user vector
to retrieve top-K candidate items per customer.

The index lives on disk at ``data/features/item_index.faiss`` between training and
inference; both training (notebook 03) and serving (``src/api/app.py``) use
:func:`save_index` / :func:`load_index`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import faiss


def build_index(item_vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Inner-product index. Caller should L2-normalize for cosine retrieval."""
    item_vectors = np.ascontiguousarray(item_vectors.astype("float32"))
    dim = item_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_vectors)
    return index


def topk(index: faiss.IndexFlatIP, user_vectors: np.ndarray, k: int = 50):
    user_vectors = np.ascontiguousarray(user_vectors.astype("float32"))
    return index.search(user_vectors, k)  # scores, ids


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x / norms


def save_index(index: faiss.IndexFlatIP, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))

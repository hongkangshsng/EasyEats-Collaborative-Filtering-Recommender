"""Similarity metrics used by the EasyEats recommender."""

from __future__ import annotations

import numpy as np


def _paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    return a[mask], b[mask]


def jaccard_similarity(a, b) -> float:
    """Jaccard similarity of items rated by both users."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    set_a = set(np.flatnonzero(~np.isnan(a)))
    set_b = set(np.flatnonzero(~np.isnan(b)))
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def cosine_similarity(a, b) -> float:
    """Cosine similarity on co-rated items."""
    x, y = _paired(a, b)
    if len(x) == 0:
        return 0.0
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else 0.0


def pearson_similarity(a, b) -> float:
    """Pearson correlation on co-rated items."""
    x, y = _paired(a, b)
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else 0.0


def rmsd_similarity(a, b) -> float:
    """Convert RMSD distance on co-rated items to (0, 1]."""
    x, y = _paired(a, b)
    if len(x) == 0:
        return 0.0
    rmsd = float(np.sqrt(np.mean((x - y) ** 2)))
    return 1.0 / (1.0 + rmsd)


METRICS = {
    "jaccard": jaccard_similarity,
    "cosine": cosine_similarity,
    "pearson": pearson_similarity,
    "rmsd": rmsd_similarity,
}

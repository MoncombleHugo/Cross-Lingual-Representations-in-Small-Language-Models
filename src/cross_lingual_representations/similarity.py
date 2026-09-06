"""Numerically stable vector normalization and cosine similarity."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


def l2_normalize(vectors: ArrayLike, *, epsilon: float = 1e-12) -> NDArray[np.float32]:
    """Return row-normalized float32 vectors without mutating cached inputs."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must have shape [items, hidden]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(norms)):
        raise ValueError("Cannot normalize non-finite representations")
    if np.any(norms <= epsilon):
        raise ValueError("Cannot normalize a zero or near-zero representation")
    return cast(NDArray[np.float32], matrix / norms)


def cosine_similarity_matrix(source: ArrayLike, target: ArrayLike) -> NDArray[np.float32]:
    """Compute all source-target cosine similarities in float32."""
    source_normalized = l2_normalize(source)
    target_normalized = l2_normalize(target)
    if source_normalized.shape[1] != target_normalized.shape[1]:
        raise ValueError("source and target hidden dimensions must match")
    return source_normalized @ target_normalized.T

"""Retrieval ranks and aggregate metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def translation_ranks(
    similarities: ArrayLike,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
) -> NDArray[np.float64]:
    """Return one-indexed translation ranks by matching sentence IDs.

    Exact ties receive their average rank, preventing identical vectors from
    producing artificially perfect retrieval.
    """
    scores = np.asarray(similarities)
    if scores.ndim != 2:
        raise ValueError("similarities must have shape [source, target]")
    if scores.shape != (len(source_ids), len(target_ids)):
        raise ValueError("similarity shape must match source_ids and target_ids")
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("source_ids and target_ids must each be unique")
    target_positions = {sentence_id: index for index, sentence_id in enumerate(target_ids)}
    missing = [sentence_id for sentence_id in source_ids if sentence_id not in target_positions]
    if missing:
        raise ValueError(f"Targets are missing {len(missing)} source sentence IDs")

    ranks = np.empty(len(source_ids), dtype=np.float64)
    for query_index, sentence_id in enumerate(source_ids):
        positive_score = scores[query_index, target_positions[sentence_id]]
        greater = np.count_nonzero(scores[query_index] > positive_score)
        tied_others = np.count_nonzero(scores[query_index] == positive_score) - 1
        ranks[query_index] = 1 + greater + tied_others / 2
    return ranks


def retrieval_metrics(ranks: ArrayLike) -> Mapping[str, float]:
    """Compute Recall@1/5/10 and mean reciprocal rank."""
    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ranks must be a non-empty one-dimensional array")
    if np.any(values < 1):
        raise ValueError("ranks must be one-indexed positive integers")
    return {
        "r1": float(np.mean(values <= 1)),
        "r5": float(np.mean(values <= 5)),
        "r10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values)),
    }

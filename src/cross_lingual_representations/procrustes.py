"""Leakage-safe orthogonal Procrustes alignment across language spaces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.retrieval import ResultRow, evaluate_direction


def _number(value: object) -> float:
    return float(cast(str | int | float, value))


def _evaluate_or_nan(
    source: ArrayLike,
    target: ArrayLike,
    sentence_ids: tuple[str, ...],
) -> tuple[dict[str, float], str]:
    source_values = np.asarray(source, dtype=np.float32)
    target_values = np.asarray(target, dtype=np.float32)
    zero_norm = np.any(np.linalg.norm(source_values, axis=1) <= 1e-12) or np.any(
        np.linalg.norm(target_values, axis=1) <= 1e-12
    )
    if zero_norm:
        return (
            {metric: float("nan") for metric in ("r1", "r5", "r10", "mrr")},
            "degenerate_zero_norm",
        )
    metrics = evaluate_direction(source_values, target_values, sentence_ids, sentence_ids)
    return dict(metrics), "ok"


@dataclass(frozen=True, slots=True)
class OrthogonalAlignment:
    """A centered source-to-target orthogonal transformation."""

    matrix: NDArray[np.float32]
    source_mean: NDArray[np.float32]
    target_mean: NDArray[np.float32]

    def transform_source(self, vectors: ArrayLike) -> NDArray[np.float32]:
        """Center and rotate source-language vectors into the target space."""
        values = np.asarray(vectors, dtype=np.float32)
        return cast(NDArray[np.float32], (values - self.source_mean) @ self.matrix)

    def center_target(self, vectors: ArrayLike) -> NDArray[np.float32]:
        """Center target-language vectors with training-only statistics."""
        return cast(
            NDArray[np.float32], np.asarray(vectors, dtype=np.float32) - self.target_mean
        )

    def transform_target_inverse(self, vectors: ArrayLike) -> NDArray[np.float32]:
        """Center and map targets back to the source space using ``W.T``."""
        values = np.asarray(vectors, dtype=np.float32)
        return cast(NDArray[np.float32], (values - self.target_mean) @ self.matrix.T)

    def center_source(self, vectors: ArrayLike) -> NDArray[np.float32]:
        """Center source-language vectors with training-only statistics."""
        return cast(
            NDArray[np.float32], np.asarray(vectors, dtype=np.float32) - self.source_mean
        )


def fit_orthogonal_procrustes(source: ArrayLike, target: ArrayLike) -> OrthogonalAlignment:
    """Fit ``argmin_W ||(X-mu_X)W-(Y-mu_Y)||`` subject to ``W.T W=I``."""
    source_values = np.asarray(source, dtype=np.float32)
    target_values = np.asarray(target, dtype=np.float32)
    if source_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("source and target must have shape [sentences, hidden]")
    if source_values.shape != target_values.shape:
        raise ValueError("source and target training matrices must have identical shapes")
    if source_values.shape[0] < 2:
        raise ValueError("At least two paired training sentences are required")
    source_mean = source_values.mean(axis=0, keepdims=True)
    target_mean = target_values.mean(axis=0, keepdims=True)
    source_centered = source_values - source_mean
    target_centered = target_values - target_mean
    cross_covariance = source_centered.T @ target_centered
    left, _singular_values, right_transpose = np.linalg.svd(
        cross_covariance, full_matrices=True
    )
    matrix = left @ right_transpose
    return OrthogonalAlignment(
        matrix=matrix.astype(np.float32, copy=False),
        source_mean=source_mean.astype(np.float32, copy=False),
        target_mean=target_mean.astype(np.float32, copy=False),
    )


def validate_train_evaluation_bundles(
    train: RepresentationBundle, evaluation: RepresentationBundle
) -> None:
    """Validate bundle compatibility and enforce split separation."""
    if train.split == evaluation.split:
        raise ValueError("Training and evaluation caches must use distinct dataset splits")
    comparable = (
        "model_name",
        "model_revision",
        "languages",
        "language_codes",
        "layer_labels",
        "pooling",
        "max_length",
    )
    mismatches = [
        field for field in comparable if getattr(train, field) != getattr(evaluation, field)
    ]
    if mismatches:
        raise ValueError(f"Training and evaluation caches differ in: {', '.join(mismatches)}")
    if train.vectors.shape[2:] != evaluation.vectors.shape[2:]:
        raise ValueError("Training and evaluation representation dimensions differ")


def evaluate_procrustes(
    train: RepresentationBundle,
    evaluation: RepresentationBundle,
) -> list[ResultRow]:
    """Fit on train and report raw/aligned/delta retrieval on evaluation."""
    validate_train_evaluation_bundles(train, evaluation)
    positions = {language: index for index, language in enumerate(train.languages)}
    num_states = len(train.layer_labels)
    depth_denominator = max(num_states - 1, 1)
    rows: list[ResultRow] = []
    for layer_index in range(num_states):
        for source_language, target_language in combinations(train.languages, 2):
            source_position = positions[source_language]
            target_position = positions[target_language]
            alignment = fit_orthogonal_procrustes(
                train.vectors[source_position, :, layer_index, :],
                train.vectors[target_position, :, layer_index, :],
            )
            directions: tuple[
                tuple[
                    str,
                    str,
                    NDArray[np.float32],
                    NDArray[np.float32],
                    NDArray[np.float32],
                    NDArray[np.float32],
                    Any,
                    Any,
                ],
                ...,
            ] = (
                (
                    source_language,
                    target_language,
                    alignment.transform_source(
                        evaluation.vectors[source_position, :, layer_index, :]
                    ),
                    alignment.center_target(
                        evaluation.vectors[target_position, :, layer_index, :]
                    ),
                    alignment.center_source(
                        evaluation.vectors[source_position, :, layer_index, :]
                    ),
                    alignment.center_target(
                        evaluation.vectors[target_position, :, layer_index, :]
                    ),
                    evaluation.vectors[source_position, :, layer_index, :],
                    evaluation.vectors[target_position, :, layer_index, :],
                ),
                (
                    target_language,
                    source_language,
                    alignment.transform_target_inverse(
                        evaluation.vectors[target_position, :, layer_index, :]
                    ),
                    alignment.center_source(
                        evaluation.vectors[source_position, :, layer_index, :]
                    ),
                    alignment.center_target(
                        evaluation.vectors[target_position, :, layer_index, :]
                    ),
                    alignment.center_source(
                        evaluation.vectors[source_position, :, layer_index, :]
                    ),
                    evaluation.vectors[target_position, :, layer_index, :],
                    evaluation.vectors[source_position, :, layer_index, :],
                ),
            )
            for (
                src_lang,
                tgt_lang,
                aligned_src,
                aligned_tgt,
                centered_src,
                centered_tgt,
                raw_src,
                raw_tgt,
            ) in directions:
                raw_metrics = evaluate_direction(
                    raw_src,
                    raw_tgt,
                    evaluation.sentence_ids,
                    evaluation.sentence_ids,
                )
                centered_metrics, centered_status = _evaluate_or_nan(
                    centered_src,
                    centered_tgt,
                    evaluation.sentence_ids,
                )
                aligned_metrics, aligned_status = _evaluate_or_nan(
                    aligned_src,
                    aligned_tgt,
                    evaluation.sentence_ids,
                )
                for metric in raw_metrics:
                    common: ResultRow = {
                        "model": evaluation.model_name,
                        "layer": layer_index,
                        "normalized_depth": layer_index / depth_denominator,
                        "pooling": evaluation.pooling,
                        "source_language": src_lang,
                        "target_language": tgt_lang,
                        "metric": metric,
                        "n_train": len(train.sentence_ids),
                        "n_eval": len(evaluation.sentence_ids),
                    }
                    raw_value = raw_metrics[metric]
                    centered_value = centered_metrics[metric]
                    aligned_value = aligned_metrics[metric]
                    rows.extend(
                        [
                            {**common, "condition": "raw", "status": "ok", "value": raw_value},
                            {
                                **common,
                                "condition": "centered",
                                "status": centered_status,
                                "value": centered_value,
                            },
                            {
                                **common,
                                "condition": "centered_delta",
                                "status": centered_status,
                                "value": centered_value - raw_value,
                            },
                            {
                                **common,
                                "condition": "aligned",
                                "status": aligned_status,
                                "value": aligned_value,
                            },
                            {
                                **common,
                                "condition": "delta",
                                "status": aligned_status,
                                "value": aligned_value - raw_value,
                            },
                        ]
                    )
    return rows


def summarize_procrustes(rows: Sequence[Mapping[str, object]]) -> list[ResultRow]:
    """Average raw, aligned, and delta metrics across directed language pairs."""
    grouped: dict[
        tuple[str, int, float, str, str, str, str, int, int], list[float]
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            int(_number(row["layer"])),
            _number(row["normalized_depth"]),
            str(row["pooling"]),
            str(row["condition"]),
            str(row["status"]),
            str(row["metric"]),
            int(_number(row["n_train"])),
            int(_number(row["n_eval"])),
        )
        grouped[key].append(_number(row["value"]))
    return [
        {
            "model": key[0],
            "layer": key[1],
            "normalized_depth": key[2],
            "pooling": key[3],
            "condition": key[4],
            "status": key[5],
            "metric": key[6],
            "value": float(np.mean(values)),
            "n_train": key[7],
            "n_eval": key[8],
            "directions": len(values),
        }
        for key, values in grouped.items()
    ]

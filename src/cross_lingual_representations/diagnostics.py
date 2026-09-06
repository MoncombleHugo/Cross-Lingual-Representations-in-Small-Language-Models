"""Geometry and uncertainty diagnostics over cached sentence representations."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch
from numpy.typing import NDArray

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.metrics import translation_ranks
from cross_lingual_representations.retrieval import ResultRow, evaluate_direction
from cross_lingual_representations.similarity import cosine_similarity_matrix, l2_normalize


def _stable_rank(values: NDArray[np.float32], *, iterations: int = 12) -> float:
    """Estimate squared Frobenius/spectral norm after centering."""
    centered = values - values.mean(axis=0, keepdims=True)
    frobenius_squared = float(np.sum(centered * centered))
    if frobenius_squared <= 1e-12:
        return 0.0
    vector = np.random.default_rng(0).normal(size=centered.shape[1]).astype(np.float32)
    vector /= np.linalg.norm(vector)
    for _ in range(iterations):
        projected = centered.T @ (centered @ vector)
        norm = np.linalg.norm(projected)
        if norm <= 1e-12:
            return 0.0
        vector = projected / norm
    spectral_squared = float(np.sum((centered @ vector) ** 2))
    return frobenius_squared / spectral_squared


def geometry_diagnostics(bundle: RepresentationBundle, *, seed: int = 42) -> list[ResultRow]:
    """Measure anisotropy, matched margins, centroids, and stable rank by layer."""
    rng = np.random.default_rng(seed)
    positions = {language: index for index, language in enumerate(bundle.languages)}
    num_states = len(bundle.layer_labels)
    rows: list[ResultRow] = []
    for layer in range(num_states):
        normalized: dict[str, NDArray[np.float32]] = {}
        for language, position in positions.items():
            values = np.asarray(bundle.vectors[position, :, layer, :], dtype=np.float32)
            unit = l2_normalize(values)
            normalized[language] = unit
            count = len(unit)
            vector_sum = unit.sum(axis=0)
            pairwise_cosine = (float(vector_sum @ vector_sum) - count) / (
                count * max(count - 1, 1)
            )
            common: ResultRow = {
                "model": bundle.model_name,
                "layer": layer,
                "normalized_depth": layer / max(num_states - 1, 1),
                "pooling": bundle.pooling,
                "language": language,
                "source_language": "",
                "target_language": "",
                "n": count,
            }
            rows.extend(
                [
                    {
                        **common,
                        "metric": "centroid_norm",
                        "value": float(np.linalg.norm(unit.mean(axis=0))),
                    },
                    {**common, "metric": "within_language_cosine", "value": pairwise_cosine},
                    {**common, "metric": "stable_rank", "value": _stable_rank(values)},
                ]
            )

        shift = int(rng.integers(1, len(bundle.sentence_ids)))
        for source_language, target_language in permutations(bundle.languages, 2):
            source = normalized[source_language]
            target = normalized[target_language]
            matched = float(np.mean(np.sum(source * target, axis=1)))
            unmatched = float(np.mean(np.sum(source * np.roll(target, shift, axis=0), axis=1)))
            common = {
                "model": bundle.model_name,
                "layer": layer,
                "normalized_depth": layer / max(num_states - 1, 1),
                "pooling": bundle.pooling,
                "language": "",
                "source_language": source_language,
                "target_language": target_language,
                "n": len(bundle.sentence_ids),
            }
            rows.extend(
                [
                    {**common, "metric": "matched_cosine", "value": matched},
                    {**common, "metric": "unmatched_cosine", "value": unmatched},
                    {**common, "metric": "cosine_margin", "value": matched - unmatched},
                ]
            )
    return rows


def bootstrap_retrieval_r1(
    bundle: RepresentationBundle,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> list[ResultRow]:
    """Bootstrap query IDs jointly across directions for direction-averaged raw R@1."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    count = len(bundle.sentence_ids)
    samples = rng.integers(0, count, size=(n_resamples, count))
    positions = {language: index for index, language in enumerate(bundle.languages)}
    rows: list[ResultRow] = []
    alpha = (1.0 - confidence) / 2.0
    for layer in range(len(bundle.layer_labels)):
        correct: list[NDArray[np.bool_]] = []
        for source_language, target_language in permutations(bundle.languages, 2):
            source = bundle.vectors[positions[source_language], :, layer, :]
            target = bundle.vectors[positions[target_language], :, layer, :]
            ranks = translation_ranks(
                cosine_similarity_matrix(source, target),
                bundle.sentence_ids,
                bundle.sentence_ids,
            )
            correct.append(ranks <= 1.0)
        correctness = np.stack(correct)
        bootstrap_values = correctness[:, samples].mean(axis=(0, 2))
        rows.append(
            {
                "model": bundle.model_name,
                "layer": layer,
                "normalized_depth": layer / max(len(bundle.layer_labels) - 1, 1),
                "pooling": bundle.pooling,
                "metric": "r1",
                "estimate": float(correctness.mean()),
                "lower": float(np.quantile(bootstrap_values, alpha)),
                "upper": float(np.quantile(bootstrap_values, 1.0 - alpha)),
                "confidence": confidence,
                "n_resamples": n_resamples,
                "n": count,
                "directions": len(correct),
            }
        )
    return rows


def query_subsample_peak_rows(
    bundle: RepresentationBundle,
    *,
    n_subsamples: int = 20,
    sample_size: int = 256,
    seed: int = 42,
) -> list[ResultRow]:
    """Measure best-layer stability over query subsets with the full candidate pool."""
    count = len(bundle.sentence_ids)
    if n_subsamples <= 0 or sample_size <= 0 or sample_size > count:
        raise ValueError("Subsample counts must be positive and sample_size cannot exceed n")
    positions = {language: index for index, language in enumerate(bundle.languages)}
    correctness: list[NDArray[np.bool_]] = []
    for layer in range(len(bundle.layer_labels)):
        by_direction = []
        for source_language, target_language in permutations(bundle.languages, 2):
            ranks = translation_ranks(
                cosine_similarity_matrix(
                    bundle.vectors[positions[source_language], :, layer, :],
                    bundle.vectors[positions[target_language], :, layer, :],
                ),
                bundle.sentence_ids,
                bundle.sentence_ids,
            )
            by_direction.append(ranks <= 1.0)
        correctness.append(np.stack(by_direction))
    values = np.stack(correctness)
    rng = np.random.default_rng(seed)
    rows: list[ResultRow] = []
    for replicate in range(n_subsamples):
        selected = rng.choice(count, size=sample_size, replace=False)
        scores = values[:, :, selected].mean(axis=(1, 2))
        best_layer = int(np.argmax(scores))
        rows.append(
            {
                "model": bundle.model_name,
                "pooling": bundle.pooling,
                "replicate": replicate,
                "sample_size": sample_size,
                "candidate_count": count,
                "best_layer": best_layer,
                "best_r1": float(scores[best_layer]),
            }
        )
    return rows


def relative_position_pool(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    *,
    layers: tuple[int, ...],
    fractions: tuple[float, ...],
) -> dict[float, torch.Tensor]:
    """Select token states at relative non-padding positions for requested layers."""
    if any(layer < 0 or layer >= len(hidden_states) for layer in layers):
        raise ValueError("Requested position-analysis layer is out of range")
    if any(fraction < 0.0 or fraction > 1.0 for fraction in fractions):
        raise ValueError("Relative positions must lie between zero and one")
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
    batch = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    result: dict[float, torch.Tensor] = {}
    for fraction in fractions:
        indices = torch.floor((lengths - 1) * fraction + 0.5).to(dtype=torch.long)
        result[fraction] = torch.stack(
            [hidden_states[layer][batch, indices] for layer in layers], dim=1
        )
    return result


def position_retrieval_rows(
    vectors: dict[float, NDArray[np.float32]],
    *,
    model_name: str,
    languages: tuple[str, ...],
    sentence_ids: tuple[str, ...],
    layers: tuple[int, ...],
) -> list[ResultRow]:
    """Evaluate relative token-position vectors over all directed language pairs."""
    positions = {language: index for index, language in enumerate(languages)}
    rows: list[ResultRow] = []
    for fraction, values in vectors.items():
        expected = (len(languages), len(sentence_ids), len(layers))
        if values.shape[:3] != expected:
            raise ValueError(f"Position vectors begin with {values.shape[:3]}, expected {expected}")
        for selected_index, layer in enumerate(layers):
            for source_language, target_language in permutations(languages, 2):
                source = values[positions[source_language], :, selected_index, :]
                target = values[positions[target_language], :, selected_index, :]
                ranks = translation_ranks(
                    cosine_similarity_matrix(source, target), sentence_ids, sentence_ids
                )
                rows.append(
                    {
                        "model": model_name,
                        "layer": layer,
                        "relative_position": fraction,
                        "source_language": source_language,
                        "target_language": target_language,
                        "metric": "r1",
                        "value": float(np.mean(ranks <= 1.0)),
                        "n": len(sentence_ids),
                    }
                )
    return rows


def block_stage_retrieval_rows(
    vectors: dict[tuple[int, str, str], NDArray[np.float32]],
    *,
    model_name: str,
    languages: tuple[str, ...],
    sentence_ids: tuple[str, ...],
) -> list[ResultRow]:
    """Evaluate pooled block input, post-attention, and output representations."""
    rows: list[ResultRow] = []
    for (block_layer, stage, pooling), values in vectors.items():
        if values.shape[:2] != (len(languages), len(sentence_ids)):
            raise ValueError("Block-stage vector language/sentence dimensions do not match")
        for source_position, target_position in permutations(range(len(languages)), 2):
            metrics = evaluate_direction(
                values[source_position],
                values[target_position],
                sentence_ids,
                sentence_ids,
            )
            for metric, value in metrics.items():
                rows.append(
                    {
                        "model": model_name,
                        "block_layer": block_layer,
                        "stage": stage,
                        "pooling": pooling,
                        "source_language": languages[source_position],
                        "target_language": languages[target_position],
                        "metric": metric,
                        "value": value,
                        "n": len(sentence_ids),
                    }
                )
    return rows

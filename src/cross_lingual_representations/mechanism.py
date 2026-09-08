"""Metrics for locating language components inside a decoder block."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations, permutations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_lingual_representations.diagnostics import stable_rank
from cross_lingual_representations.language_subspace import centroid_language_subspace
from cross_lingual_representations.retrieval import ResultRow, evaluate_direction
from cross_lingual_representations.similarity import l2_normalize

StageVectors = Mapping[tuple[str, str], NDArray[np.float32]]


def _probe_matrix(
    vectors: NDArray[np.float32], languages: Sequence[str]
) -> tuple[NDArray[np.float32], NDArray[np.str_]]:
    language_count, sentence_count, hidden_size = vectors.shape
    return (
        vectors.reshape(language_count * sentence_count, hidden_size),
        np.repeat(np.asarray(languages), sentence_count),
    )


def _anisotropy(values: NDArray[np.float32]) -> float:
    by_language: list[float] = []
    for language_values in values:
        unit = l2_normalize(language_values)
        count = len(unit)
        vector_sum = unit.sum(axis=0)
        by_language.append((float(vector_sum @ vector_sum) - count) / (count * max(count - 1, 1)))
    return float(np.mean(by_language))


def evaluate_block_mechanism(
    train_vectors: StageVectors,
    evaluation_vectors: StageVectors,
    *,
    model_name: str,
    block_layer: int,
    languages: tuple[str, ...],
    train_ids: tuple[str, ...],
    evaluation_ids: tuple[str, ...],
    subspace_dimensions: int = 3,
    c: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> list[ResultRow]:
    """Measure retrieval, language identity, and geometry at three block stages."""
    if set(train_vectors) != set(evaluation_vectors):
        raise ValueError("Train and evaluation stage keys must match")
    if not 0 < subspace_dimensions <= len(languages) - 1:
        raise ValueError("subspace_dimensions must be between 1 and languages - 1")
    if c <= 0 or max_iter <= 0:
        raise ValueError("c and max_iter must be positive")
    rows: list[ResultRow] = []
    for stage, pooling in sorted(train_vectors):
        train = np.asarray(train_vectors[(stage, pooling)], dtype=np.float32)
        evaluation = np.asarray(evaluation_vectors[(stage, pooling)], dtype=np.float32)
        if train.shape[:2] != (len(languages), len(train_ids)):
            raise ValueError("Train stage vectors do not match language/sample metadata")
        if evaluation.shape[:2] != (len(languages), len(evaluation_ids)):
            raise ValueError("Evaluation stage vectors do not match language/sample metadata")
        if train.shape[-1] != evaluation.shape[-1]:
            raise ValueError("Train and evaluation hidden sizes differ")

        centroids = train.mean(axis=1)
        global_centroid = centroids.mean(axis=0)
        language_basis, explained = centroid_language_subspace(train)
        basis = language_basis[:, :subspace_dimensions].astype(np.float32)
        centered_evaluation = evaluation - centroids[:, None, :]
        flat_evaluation = evaluation.reshape(-1, evaluation.shape[-1])
        globally_centered = flat_evaluation - global_centroid
        projected = globally_centered @ basis
        total_variance = float(np.sum(globally_centered * globally_centered))
        language_variance = float(np.sum(projected * projected))
        separation = float(
            np.mean(
                [
                    np.linalg.norm(centroids[left] - centroids[right])
                    for left, right in combinations(range(len(languages)), 2)
                ]
            )
        )
        common: ResultRow = {
            "model": model_name,
            "block_layer": block_layer,
            "stage": stage,
            "pooling": pooling,
            "condition": "geometry",
            "source_language": "",
            "target_language": "",
            "metric": "",
            "value": 0.0,
            "k": subspace_dimensions,
            "n_train": len(train_ids) * len(languages),
            "n_eval": len(evaluation_ids) * len(languages),
        }
        geometry = {
            "centroid_separation": separation,
            "centroid_norm": float(np.mean(np.linalg.norm(centroids, axis=1))),
            "language_subspace_variance_fraction": (
                language_variance / total_variance if total_variance > 0 else 0.0
            ),
            "language_subspace_centroid_variance": float(explained[:subspace_dimensions].sum()),
            "anisotropy": _anisotropy(evaluation),
            "stable_rank": stable_rank(np.asarray(globally_centered, dtype=np.float32)),
        }
        rows.extend(
            {**common, "metric": metric, "value": value} for metric, value in geometry.items()
        )

        for condition, values in (
            ("raw", evaluation),
            ("per-language-centered", centered_evaluation),
        ):
            for source, target in permutations(range(len(languages)), 2):
                metrics = evaluate_direction(
                    values[source], values[target], evaluation_ids, evaluation_ids
                )
                for metric in ("r1", "mrr"):
                    rows.append(
                        {
                            **common,
                            "condition": condition,
                            "source_language": languages[source],
                            "target_language": languages[target],
                            "metric": metric,
                            "value": metrics[metric],
                        }
                    )

        train_features, train_labels = _probe_matrix(train, languages)
        eval_features, eval_labels = _probe_matrix(evaluation, languages)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                max_iter=max_iter,
                random_state=random_state,
                solver="lbfgs",
            ),
        )
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(eval_features)
        for metric, value in (
            ("language_probe_accuracy", accuracy_score(eval_labels, predictions)),
            ("language_probe_macro_f1", f1_score(eval_labels, predictions, average="macro")),
        ):
            rows.append(
                {
                    **common,
                    "condition": "raw",
                    "metric": metric,
                    "value": float(value),
                }
            )
    return rows

"""Identify and remove low-dimensional language-specific directions."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.procrustes import validate_train_evaluation_bundles
from cross_lingual_representations.retrieval import ResultRow, evaluate_direction


def centroid_language_subspace(
    vectors: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return centroid-difference right singular vectors and variance fractions."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("vectors must have shape [languages >= 2, samples, hidden]")
    centroids = values.mean(axis=1)
    differences = centroids - centroids.mean(axis=0, keepdims=True)
    _left, singular_values, right = np.linalg.svd(differences, full_matrices=False)
    variance = singular_values**2
    total = float(variance.sum())
    explained = variance / total if total > 0 else np.zeros_like(variance)
    return right.T, explained


def project_out(
    vectors: NDArray[np.floating[Any]], basis: NDArray[np.floating[Any]]
) -> NDArray[np.float32]:
    """Remove the orthogonal projection onto the columns of ``basis``."""
    values = np.asarray(vectors, dtype=np.float32)
    directions = np.asarray(basis, dtype=np.float32)
    if directions.ndim != 2 or directions.shape[0] != values.shape[-1]:
        raise ValueError("basis must have shape [hidden, dimensions]")
    return np.asarray(values - (values @ directions) @ directions.T, dtype=np.float32)


def random_orthonormal_basis(hidden_size: int, dimensions: int, seed: int) -> NDArray[np.float64]:
    """Generate a reproducible random orthonormal control basis."""
    if not 0 < dimensions <= hidden_size:
        raise ValueError("dimensions must be between 1 and hidden_size")
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(hidden_size, dimensions))
    basis, _upper = np.linalg.qr(matrix, mode="reduced")
    return basis


def _probe_features(
    vectors: NDArray[np.floating[Any]], languages: Sequence[str]
) -> tuple[NDArray[np.float32], NDArray[np.str_]]:
    language_count, sentence_count, hidden_size = vectors.shape
    features = np.asarray(vectors, dtype=np.float32).reshape(
        language_count * sentence_count, hidden_size
    )
    labels = np.repeat(np.asarray(languages), sentence_count)
    return features, labels


def _condition_bases(
    train_vectors: NDArray[np.floating[Any]],
    dimensions: Sequence[int],
    random_seeds: Sequence[int],
    language_basis: NDArray[np.float64],
    *,
    layer: int,
) -> list[tuple[str, int, int | None, NDArray[np.float64] | None]]:
    max_dimension = max(dimensions)
    flattened = np.asarray(train_vectors, dtype=np.float32).reshape(-1, train_vectors.shape[-1])
    pca = PCA(n_components=max_dimension, svd_solver="randomized", random_state=layer)
    pca.fit(flattened)
    pca_basis = np.asarray(pca.components_.T, dtype=np.float64)
    conditions: list[tuple[str, int, int | None, NDArray[np.float64] | None]] = [
        ("raw", 0, None, None)
    ]
    for dimension in dimensions:
        conditions.append(("language_subspace", dimension, None, language_basis[:, :dimension]))
        conditions.append(("top_pca", dimension, None, pca_basis[:, :dimension]))
        for seed in random_seeds:
            random_basis = random_orthonormal_basis(
                train_vectors.shape[-1], max_dimension, seed + 10_000 * layer
            )
            conditions.append(("random", dimension, seed, random_basis[:, :dimension]))
    return conditions


def evaluate_language_subspace(
    train: RepresentationBundle,
    evaluation: RepresentationBundle,
    *,
    dimensions: Sequence[int] = (1, 2, 3),
    random_seeds: Sequence[int] = (42,),
    c: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> list[ResultRow]:
    """Evaluate variance, translation retrieval, and language probes after removal."""
    validate_train_evaluation_bundles(train, evaluation)
    if not dimensions or max(dimensions) > len(train.languages) - 1:
        raise ValueError("dimensions exceed the centroid subspace rank")
    if c <= 0 or max_iter <= 0:
        raise ValueError("c and max_iter must be positive")
    rows: list[ResultRow] = []
    states = len(train.layer_labels)
    language_positions = {language: index for index, language in enumerate(train.languages)}
    for layer in range(states):
        normalized_depth = layer / max(states - 1, 1)
        train_vectors = np.asarray(train.vectors[:, :, layer, :], dtype=np.float32)
        eval_vectors = np.asarray(evaluation.vectors[:, :, layer, :], dtype=np.float32)
        language_basis, explained = centroid_language_subspace(train_vectors)
        common: ResultRow = {
            "model": evaluation.model_name,
            "layer": layer,
            "normalized_depth": normalized_depth,
            "pooling": evaluation.pooling,
            "record_type": "subspace_variance",
            "basis": "language_subspace",
            "k": 0,
            "seed": "",
            "source_language": "",
            "target_language": "",
            "metric": "cumulative_explained_variance",
            "value": 0.0,
            "n_train": len(train.sentence_ids) * len(train.languages),
            "n_eval": len(evaluation.sentence_ids) * len(evaluation.languages),
        }
        for dimension in dimensions:
            rows.append(
                {
                    **common,
                    "k": dimension,
                    "value": float(explained[:dimension].sum()),
                }
            )

        conditions = _condition_bases(
            train_vectors,
            dimensions,
            random_seeds,
            language_basis,
            layer=layer,
        )
        for basis_name, dimension, seed, basis in conditions:
            transformed_train = (
                train_vectors if basis is None else project_out(train_vectors, basis)
            )
            transformed_eval = eval_vectors if basis is None else project_out(eval_vectors, basis)
            condition_common: ResultRow = {
                **common,
                "basis": basis_name,
                "k": dimension,
                "seed": "" if seed is None else seed,
            }
            for source_language, target_language in permutations(train.languages, 2):
                source = transformed_eval[language_positions[source_language]]
                target = transformed_eval[language_positions[target_language]]
                retrieval = evaluate_direction(
                    source,
                    target,
                    evaluation.sentence_ids,
                    evaluation.sentence_ids,
                )
                for metric in ("r1", "mrr"):
                    rows.append(
                        {
                            **condition_common,
                            "record_type": "retrieval",
                            "source_language": source_language,
                            "target_language": target_language,
                            "metric": metric,
                            "value": retrieval[metric],
                        }
                    )

            train_features, train_labels = _probe_features(transformed_train, train.languages)
            eval_features, eval_labels = _probe_features(transformed_eval, evaluation.languages)
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
                ("accuracy", accuracy_score(eval_labels, predictions)),
                ("macro_f1", f1_score(eval_labels, predictions, average="macro")),
            ):
                rows.append(
                    {
                        **condition_common,
                        "record_type": "language_probe",
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows

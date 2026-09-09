"""Evaluation utilities for residual-stream causal interventions."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_lingual_representations.language_subspace import centroid_language_subspace
from cross_lingual_representations.metrics import translation_ranks
from cross_lingual_representations.retrieval import ResultRow, evaluate_direction
from cross_lingual_representations.similarity import cosine_similarity_matrix

InterventionKey = tuple[str, int | None, str]
InterventionVectors = dict[InterventionKey, NDArray[np.float32]]


def intervention_bases(
    train_vectors: NDArray[np.floating], dimensions: int
) -> dict[str, NDArray[np.float32]]:
    """Learn language-centroid and label-free PCA bases from train vectors only."""
    values = np.asarray(train_vectors, dtype=np.float32)
    if values.ndim != 3 or not 0 < dimensions <= min(values.shape[0] - 1, values.shape[-1]):
        raise ValueError("train_vectors must be [languages, samples, hidden] with a valid k")
    language, _explained = centroid_language_subspace(values)
    flattened = values.reshape(-1, values.shape[-1]).astype(np.float64)
    flattened -= flattened.mean(axis=0, keepdims=True)
    _left, _singular, right = np.linalg.svd(flattened, full_matrices=False)
    bases = {
        "language": np.asarray(language[:, :dimensions], dtype=np.float32),
        "pca": np.asarray(right[:dimensions].T, dtype=np.float32),
    }
    for name, basis in bases.items():
        validate_intervention_basis(basis, name=name)
    return bases


def validate_intervention_basis(
    basis: NDArray[np.floating], *, name: str = "basis", atol: float = 2e-5
) -> None:
    """Check orthonormality and the float32 projection invariant."""
    directions = np.asarray(basis, dtype=np.float32)
    if directions.ndim != 2 or not np.isfinite(directions).all():
        raise ValueError(f"{name} must be a finite rank-2 array")
    identity = np.eye(directions.shape[1], dtype=np.float32)
    if not np.allclose(directions.T @ directions, identity, atol=atol, rtol=atol):
        raise ValueError(f"{name} is not orthonormal")
    probe = np.arange(2 * directions.shape[0], dtype=np.float32).reshape(
        2, directions.shape[0]
    )
    projected = probe - (probe @ directions) @ directions.T
    residual = projected @ directions
    scale = max(1.0, float(np.abs(probe).max()))
    if float(np.abs(residual).max()) > 10 * atol * scale:
        raise ValueError(f"{name} projection invariant failed")


def paired_retrieval_bootstrap(
    evaluation_vectors: InterventionVectors,
    *,
    languages: tuple[str, ...],
    layer_indices: tuple[int, ...],
    comparisons: tuple[tuple[str, str], ...],
    n_resamples: int = 1000,
    random_state: int = 42,
) -> list[ResultRow]:
    """Bootstrap paired final-layer R@1 deltas over shared FLORES sentence IDs."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    final_position = len(layer_indices) - 1
    final_layer = layer_indices[final_position]
    rows: list[ResultRow] = []
    poolings = sorted({key[2] for key in evaluation_vectors})
    for pooling in poolings:
        available = {
            condition: values
            for (condition, seed, key_pooling), values in evaluation_vectors.items()
            if key_pooling == pooling and seed is None
        }
        correctness: dict[str, NDArray[np.float64]] = {}
        for condition, values in available.items():
            directional = []
            for source, target in permutations(range(len(languages)), 2):
                similarities = cosine_similarity_matrix(
                    values[source, :, final_position, :],
                    values[target, :, final_position, :],
                )
                ids = tuple(str(index) for index in range(values.shape[1]))
                ranks = translation_ranks(similarities, ids, ids)
                directional.append(np.asarray(ranks <= 1.0, dtype=np.float64))
            correctness[condition] = np.stack(directional)
        sentence_count = next(iter(correctness.values())).shape[1]
        generator = np.random.default_rng(random_state)
        samples = generator.integers(0, sentence_count, size=(n_resamples, sentence_count))
        for left, right in comparisons:
            if left not in correctness or right not in correctness:
                raise ValueError(f"Missing bootstrap condition: {left} or {right}")
            paired = correctness[left] - correctness[right]
            observed = float(paired.mean())
            values = paired[:, samples].mean(axis=(0, 2))
            rows.append(
                {
                    "model": "",
                    "layer": final_layer,
                    "normalized_depth": 1.0,
                    "pooling": pooling,
                    "condition": left,
                    "seed": "",
                    "source_language": "",
                    "target_language": "",
                    "metric": "delta_r1",
                    "value": observed,
                    "n_train": 0,
                    "n_eval": sentence_count * len(languages),
                    "record_type": "bootstrap_delta",
                    "comparison": f"{left}-{right}",
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "n_resamples": n_resamples,
                }
            )
    return rows


def evaluate_residual_interventions(
    train_vectors: InterventionVectors,
    evaluation_vectors: InterventionVectors,
    *,
    model_name: str,
    languages: tuple[str, ...],
    train_ids: tuple[str, ...],
    evaluation_ids: tuple[str, ...],
    layer_indices: tuple[int, ...],
    total_layers: int,
    c: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
    include_language_probe: bool = True,
) -> list[ResultRow]:
    """Measure later-layer retrieval and language identity after interventions."""
    if set(train_vectors) != set(evaluation_vectors):
        raise ValueError("Train and evaluation intervention keys must match")
    if not layer_indices or total_layers < max(layer_indices):
        raise ValueError("Invalid intervention layer indices")
    rows: list[ResultRow] = []
    for condition, seed, pooling in sorted(
        train_vectors, key=lambda key: (key[0], -1 if key[1] is None else key[1], key[2])
    ):
        train = np.asarray(train_vectors[(condition, seed, pooling)], dtype=np.float32)
        evaluation = np.asarray(evaluation_vectors[(condition, seed, pooling)], dtype=np.float32)
        if train.shape[:3] != (len(languages), len(train_ids), len(layer_indices)):
            raise ValueError("Train intervention vectors do not match metadata")
        if evaluation.shape[:3] != (
            len(languages),
            len(evaluation_ids),
            len(layer_indices),
        ):
            raise ValueError("Evaluation intervention vectors do not match metadata")
        for state_position, layer in enumerate(layer_indices):
            common: ResultRow = {
                "model": model_name,
                "layer": layer,
                "normalized_depth": layer / max(total_layers, 1),
                "pooling": pooling,
                "condition": condition,
                "seed": "" if seed is None else seed,
                "source_language": "",
                "target_language": "",
                "metric": "",
                "value": 0.0,
                "n_train": len(train_ids) * len(languages),
                "n_eval": len(evaluation_ids) * len(languages),
            }
            for source, target in permutations(range(len(languages)), 2):
                metrics = evaluate_direction(
                    evaluation[source, :, state_position, :],
                    evaluation[target, :, state_position, :],
                    evaluation_ids,
                    evaluation_ids,
                )
                for metric in ("r1", "mrr"):
                    rows.append(
                        {
                            **common,
                            "source_language": languages[source],
                            "target_language": languages[target],
                            "metric": metric,
                            "value": metrics[metric],
                        }
                    )
            if include_language_probe:
                train_features = train[:, :, state_position, :].reshape(
                    len(languages) * len(train_ids), -1
                )
                evaluation_features = evaluation[:, :, state_position, :].reshape(
                    len(languages) * len(evaluation_ids), -1
                )
                train_labels = np.repeat(np.asarray(languages), len(train_ids))
                evaluation_labels = np.repeat(np.asarray(languages), len(evaluation_ids))
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
                predictions = classifier.predict(evaluation_features)
                for metric, value in (
                    ("language_probe_accuracy", accuracy_score(evaluation_labels, predictions)),
                    (
                        "language_probe_macro_f1",
                        f1_score(evaluation_labels, predictions, average="macro"),
                    ),
                ):
                    rows.append({**common, "metric": metric, "value": float(value)})
    return rows

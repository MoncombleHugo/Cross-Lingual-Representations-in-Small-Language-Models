"""Evaluation utilities for residual-stream causal interventions."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_lingual_representations.retrieval import ResultRow, evaluate_direction

InterventionKey = tuple[str, int | None, str]
InterventionVectors = dict[InterventionKey, NDArray[np.float32]]


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

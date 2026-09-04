"""Linear language-identity probes over cached sentence representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.procrustes import validate_train_evaluation_bundles
from cross_lingual_representations.retrieval import ResultRow


@dataclass(frozen=True, slots=True)
class ProbeEvaluation:
    """Layer metrics plus selected-layer confusion matrices."""

    metrics: list[ResultRow]
    confusion: list[ResultRow]


def _probe_matrix(
    bundle: RepresentationBundle, layer: int
) -> tuple[NDArray[np.float32], NDArray[np.str_]]:
    language_count, sentence_count, _state_count, hidden_size = bundle.vectors.shape
    features = np.asarray(bundle.vectors[:, :, layer, :], dtype=np.float32).reshape(
        language_count * sentence_count, hidden_size
    )
    labels = np.repeat(np.asarray(bundle.languages), sentence_count)
    return features, labels


def evaluate_language_probe(
    train: RepresentationBundle,
    evaluation: RepresentationBundle,
    *,
    c: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> ProbeEvaluation:
    """Fit one standardized multinomial linear classifier per hidden state."""
    validate_train_evaluation_bundles(train, evaluation)
    if c <= 0 or max_iter <= 0:
        raise ValueError("c and max_iter must be positive")
    num_states = len(train.layer_labels)
    representative_layers = {0, num_states // 2, num_states - 1}
    metric_rows: list[ResultRow] = []
    confusion_rows: list[ResultRow] = []
    for layer in range(num_states):
        train_features, train_labels = _probe_matrix(train, layer)
        eval_features, eval_labels = _probe_matrix(evaluation, layer)
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
        common: ResultRow = {
            "model": evaluation.model_name,
            "layer": layer,
            "normalized_depth": layer / max(num_states - 1, 1),
            "pooling": evaluation.pooling,
            "n_train": len(train_labels),
            "n_eval": len(eval_labels),
            "n_classes": len(train.languages),
        }
        metric_rows.extend(
            [
                {
                    **common,
                    "metric": "accuracy",
                    "value": float(accuracy_score(eval_labels, predictions)),
                },
                {
                    **common,
                    "metric": "macro_f1",
                    "value": float(f1_score(eval_labels, predictions, average="macro")),
                },
            ]
        )
        if layer in representative_layers:
            matrix = confusion_matrix(eval_labels, predictions, labels=train.languages)
            for true_index, true_language in enumerate(train.languages):
                for predicted_index, predicted_language in enumerate(train.languages):
                    confusion_rows.append(
                        {
                            **common,
                            "true_language": true_language,
                            "predicted_language": predicted_language,
                            "count": int(matrix[true_index, predicted_index]),
                        }
                    )
    return ProbeEvaluation(metric_rows, confusion_rows)

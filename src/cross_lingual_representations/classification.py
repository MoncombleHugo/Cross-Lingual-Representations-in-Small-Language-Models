"""Leakage-safe English-only multilingual intent transfer on frozen representations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.retrieval import ResultRow


@dataclass(frozen=True, slots=True)
class ClassificationSplit:
    """One language split with labels kept outside the representation cache."""

    split: str
    language: str
    language_code: str
    ids: tuple[str, ...]
    texts: tuple[str, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.texts) or len(self.ids) != len(self.labels):
            raise ValueError("Classification IDs, texts, and labels must be equally sized")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Classification IDs must be unique")
        if any(not text.strip() for text in self.texts):
            raise ValueError("Classification texts must be non-empty")

    @property
    def sentences(self) -> Mapping[str, tuple[str, ...]]:
        return {self.language: self.texts}

    @property
    def language_codes(self) -> Mapping[str, str]:
        return {self.language: self.language_code}


def sample_classification_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str,
    language: str,
    language_code: str,
    n_samples: int,
    seed: int,
    text_field: str = "utt",
    label_field: str = "intent",
    id_field: str = "id",
) -> ClassificationSplit:
    """Take a deterministic stratified sample while preserving source IDs."""
    materialized = list(records)
    if n_samples <= 0 or n_samples > len(materialized):
        raise ValueError(f"Requested {n_samples} rows from {len(materialized)} examples")
    labels = np.asarray([str(row[label_field]) for row in materialized])
    unique_labels, counts = np.unique(labels, return_counts=True)
    if n_samples < len(unique_labels):
        raise ValueError("n_samples must be at least the number of classes")
    desired = counts.astype(np.float64) * n_samples / len(materialized)
    allocations = np.minimum(counts, np.maximum(1, np.floor(desired).astype(int)))
    while int(allocations.sum()) < n_samples:
        candidates = np.flatnonzero(allocations < counts)
        if not len(candidates):
            raise RuntimeError("Unable to allocate the requested stratified sample")
        priority = desired[candidates] - allocations[candidates]
        chosen = int(candidates[int(np.argmax(priority))])
        allocations[chosen] += 1
    while int(allocations.sum()) > n_samples:
        candidates = np.flatnonzero(allocations > 1)
        priority = allocations[candidates] - desired[candidates]
        chosen = int(candidates[int(np.argmax(priority))])
        allocations[chosen] -= 1
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for label, allocation in zip(unique_labels, allocations, strict=True):
        label_indices = np.flatnonzero(labels == label)
        selected_indices.extend(
            int(index)
            for index in rng.choice(label_indices, int(allocation), replace=False)
        )
    indices = np.asarray(sorted(selected_indices), dtype=np.intp)
    ids: list[str] = []
    texts: list[str] = []
    selected_labels: list[str] = []
    for index in indices:
        row = materialized[int(index)]
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid classification text at row {int(index)}")
        if id_field not in row:
            raise ValueError(f"Classification row is missing {id_field!r}")
        ids.append(str(row[id_field]))
        texts.append(text.strip())
        selected_labels.append(str(row[label_field]))
    return ClassificationSplit(
        split=split,
        language=language,
        language_code=language_code,
        ids=tuple(ids),
        texts=tuple(texts),
        labels=tuple(selected_labels),
    )


def load_classification_split(
    *,
    dataset_name: str,
    dataset_config: str,
    source_split: str,
    language: str,
    language_code: str,
    n_samples: int,
    seed: int,
    text_field: str,
    label_field: str,
    id_field: str,
    data_file_template: str | None = None,
    load_dataset_fn: Any | None = None,
) -> ClassificationSplit:
    """Load one locale independently from the public MASSIVE parquet data."""
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset
    if data_file_template is None:
        records = load_dataset_fn(dataset_name, dataset_config, split=source_split)
    else:
        data_file = data_file_template.format(
            dataset=dataset_name,
            config=dataset_config,
            split=source_split,
        )
        records = load_dataset_fn(
            "parquet", data_files={source_split: data_file}, split=source_split
        )
    return sample_classification_rows(
        records,
        split=f"classification:{dataset_name}:{dataset_config}:{source_split}",
        language=language,
        language_code=language_code,
        n_samples=n_samples,
        seed=seed,
        text_field=text_field,
        label_field=label_field,
        id_field=id_field,
    )


def layer_number(label: str) -> int:
    """Recover the original hidden-state number from a canonical cache label."""
    if not label.startswith("layer_"):
        raise ValueError(f"Unrecognized layer label: {label!r}")
    return int(label.split("_", maxsplit=2)[1])


def resolve_transfer_layers(
    retrieval_rows: Sequence[Mapping[str, object]],
    *,
    pooling: str,
    selectors: Sequence[str | int],
) -> tuple[int, ...]:
    """Resolve configured layer aliases, including the best raw FLORES layer."""
    raw = [
        row
        for row in retrieval_rows
        if row["pooling"] == pooling
        and row["condition"] == "raw"
        and row["metric"] == "r1"
    ]
    if not raw:
        raise ValueError(f"No raw FLORES retrieval rows for pooling {pooling!r}")
    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in raw:
        by_layer[int(cast(str, row["layer"]))].append(float(cast(str, row["value"])))
    final_layer = max(by_layer)
    best_raw = max(by_layer, key=lambda layer: float(np.mean(by_layer[layer])))
    aliases = {
        "embedding": 0,
        "best_raw": best_raw,
        "intermediate": final_layer // 2,
        "final": final_layer,
    }
    resolved = [
        aliases[selector] if isinstance(selector, str) else selector
        for selector in selectors
    ]
    if any(layer not in by_layer for layer in resolved):
        raise ValueError("A requested downstream layer is absent from FLORES retrieval")
    return tuple(dict.fromkeys(resolved))


def _validate_transfer_bundles(
    train: Mapping[str, RepresentationBundle],
    evaluation: Mapping[str, RepresentationBundle],
    labels: Mapping[str, tuple[str, ...]],
) -> None:
    if set(train) != set(evaluation) or set(train) != set(labels):
        raise ValueError("Train, evaluation, and label languages must match")
    reference = next(iter(train.values()))
    for language in train:
        train_bundle = train[language]
        eval_bundle = evaluation[language]
        for field in ("model_name", "model_revision", "pooling", "layer_labels"):
            if getattr(train_bundle, field) != getattr(reference, field):
                raise ValueError(f"Training bundle mismatch in {field}")
            if getattr(eval_bundle, field) != getattr(reference, field):
                raise ValueError(f"Evaluation bundle mismatch in {field}")
        if train_bundle.languages != (language,) or eval_bundle.languages != (language,):
            raise ValueError("Every classification cache must contain exactly its named language")
        if train_bundle.split == eval_bundle.split or set(train_bundle.sentence_ids) & set(
            eval_bundle.sentence_ids
        ):
            raise ValueError("Classification train and evaluation examples must be disjoint")
        if len(eval_bundle.sentence_ids) != len(labels[language]):
            raise ValueError(f"Evaluation labels do not match cached rows for {language}")


def evaluate_crosslingual_transfer(
    train: Mapping[str, RepresentationBundle],
    evaluation: Mapping[str, RepresentationBundle],
    train_labels: tuple[str, ...],
    evaluation_labels: Mapping[str, tuple[str, ...]],
    *,
    layers: Sequence[int],
    train_language: str = "en",
    c: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> list[ResultRow]:
    """Train on English only and evaluate raw/global/per-language centered features."""
    _validate_transfer_bundles(train, evaluation, evaluation_labels)
    if len(train[train_language].sentence_ids) != len(train_labels):
        raise ValueError("English training labels do not match cached rows")
    label_set = set(train_labels)
    if any(set(values) - label_set for values in evaluation_labels.values()):
        raise ValueError("Evaluation contains a label unseen in English training")
    label_to_position = {
        layer_number(label): position
        for position, label in enumerate(train[train_language].layer_labels)
    }
    if any(layer not in label_to_position for layer in layers):
        raise ValueError("A requested downstream layer is absent from the cache")
    final_layer = max(label_to_position)
    rows: list[ResultRow] = []
    for layer in layers:
        position = label_to_position[layer]
        means = {
            language: np.asarray(bundle.vectors[0, :, position, :], dtype=np.float32).mean(
                axis=0
            )
            for language, bundle in train.items()
        }
        global_mean = np.mean(np.stack(list(means.values()), axis=0), axis=0)
        english_train = np.asarray(
            train[train_language].vectors[0, :, position, :], dtype=np.float32
        )
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                max_iter=max_iter,
                random_state=random_state,
                solver="lbfgs",
            ),
        )
        classifier.fit(english_train, np.asarray(train_labels))
        scores: dict[tuple[str, str], tuple[float, float]] = {}
        for condition in ("raw", "global-centered", "per-language-centered"):
            for language, bundle in evaluation.items():
                features = np.asarray(bundle.vectors[0, :, position, :], dtype=np.float32)
                if condition == "per-language-centered":
                    # Algebraically equivalent to fitting StandardScaler on English-centered train.
                    features = features + means[train_language] - means[language]
                elif condition == "global-centered":
                    features = features + global_mean - global_mean
                predictions = classifier.predict(features)
                truth = np.asarray(evaluation_labels[language])
                scores[(condition, language)] = (
                    float(accuracy_score(truth, predictions)),
                    float(f1_score(truth, predictions, average="macro")),
                )
        for condition in ("raw", "global-centered", "per-language-centered"):
            english_scores = scores[(condition, train_language)]
            for language in evaluation:
                accuracy, macro_f1 = scores[(condition, language)]
                for metric, value, english_score in (
                    ("accuracy", accuracy, english_scores[0]),
                    ("macro_f1", macro_f1, english_scores[1]),
                ):
                    rows.append(
                        {
                            "model": train[train_language].model_name,
                            "layer": layer,
                            "normalized_depth": layer / max(final_layer, 1),
                            "pooling": train[train_language].pooling,
                            "condition": condition,
                            "train_language": train_language,
                            "test_language": language,
                            "metric": metric,
                            "value": value,
                            "accuracy": accuracy,
                            "macro_f1": macro_f1,
                            "english_score": english_score,
                            "transfer_gap": english_score - value,
                            "n_train": len(train_labels),
                            "n_centroid_samples": len(train[language].sentence_ids),
                            "n_eval": len(evaluation_labels[language]),
                            "n_classes": len(label_set),
                        }
                    )
    return rows


def retrieval_transfer_gain_rows(
    transfer_rows: Sequence[Mapping[str, object]],
    retrieval_rows: Sequence[Mapping[str, object]],
) -> list[ResultRow]:
    """Relate target-specific FLORES centering gains to downstream transfer gains."""
    points: list[ResultRow] = []
    models = {str(row["model"]) for row in transfer_rows}
    for model in models:
        for metric in ("accuracy", "macro_f1"):
            selected = [
                row
                for row in transfer_rows
                if row["model"] == model
                and row["metric"] == metric
                and row["test_language"] != row["train_language"]
            ]
            keys = {
                (str(row["pooling"]), int(cast(int, row["layer"])), str(row["test_language"]))
                for row in selected
            }
            metric_points: list[ResultRow] = []
            for pooling, layer, language in sorted(keys):
                transfer_values = {
                    str(row["condition"]): float(cast(float, row["value"]))
                    for row in selected
                    if row["pooling"] == pooling
                    and int(cast(int, row["layer"])) == layer
                    and row["test_language"] == language
                }
                relevant_retrieval = [
                    row
                    for row in retrieval_rows
                    if row["model"] == model
                    and row["pooling"] == pooling
                    and int(cast(str, row["layer"])) == layer
                    and row["metric"] == "r1"
                    and row["target_language"] == language
                    and row.get("status", "ok") == "ok"
                ]
                maximum_samples = max(
                    int(cast(str, row["n_centroid_samples"]))
                    for row in relevant_retrieval
                    if row["centering_condition"] == "nonparallel-centered"
                )
                retrieval_raw = np.mean(
                    [
                        float(cast(str, row["value"]))
                        for row in relevant_retrieval
                        if row["centering_condition"] == "raw"
                    ]
                )
                retrieval_centered = np.mean(
                    [
                        float(cast(str, row["value"]))
                        for row in relevant_retrieval
                        if row["centering_condition"] == "nonparallel-centered"
                        and int(cast(str, row["n_centroid_samples"])) == maximum_samples
                    ]
                )
                transfer_raw = transfer_values["raw"]
                transfer_centered = transfer_values["per-language-centered"]
                metric_points.append(
                    {
                        "model": model,
                        "pooling": pooling,
                        "layer": layer,
                        "target_language": language,
                        "metric": metric,
                        "retrieval_raw": float(retrieval_raw),
                        "retrieval_centered": float(retrieval_centered),
                        "retrieval_gain": float(retrieval_centered - retrieval_raw),
                        "transfer_raw": transfer_raw,
                        "transfer_centered": transfer_centered,
                        "transfer_gain": transfer_centered - transfer_raw,
                        "pearson_r": float("nan"),
                        "spearman_r": float("nan"),
                    }
                )
            if len(metric_points) >= 2:
                retrieval_gains = [float(row["retrieval_gain"]) for row in metric_points]
                transfer_gains = [float(row["transfer_gain"]) for row in metric_points]
                pearson = float(pearsonr(retrieval_gains, transfer_gains).statistic)
                spearman = float(spearmanr(retrieval_gains, transfer_gains).statistic)
                for row in metric_points:
                    row["pearson_r"] = pearson
                    row["spearman_r"] = spearman
            points.extend(metric_points)
    return points

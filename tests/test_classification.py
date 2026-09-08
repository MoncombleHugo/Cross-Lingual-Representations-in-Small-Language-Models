from __future__ import annotations

from typing import Any

import numpy as np

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.classification import (
    evaluate_crosslingual_transfer,
    resolve_transfer_layers,
    sample_classification_rows,
)


def _bundle(
    values: np.ndarray[Any, Any], *, language: str, split: str
) -> RepresentationBundle:
    return RepresentationBundle(
        vectors=values[None, :, None, :].astype(np.float32),
        model_name="model",
        model_revision=None,
        split=split,
        languages=(language,),
        language_codes=(language,),
        sentence_ids=tuple(f"{split}-{index}" for index in range(len(values))),
        layer_labels=("layer_24",),
        pooling="mean",
        seed=42,
        max_length=64,
        inference_dtype="float32",
        representation_dtype="float32",
        created_at="now",
    )


def test_classification_sampling_is_stratified_and_reproducible() -> None:
    rows = [
        {"id": index, "utt": f"text {index}", "intent": index % 3}
        for index in range(30)
    ]
    first = sample_classification_rows(
        rows,
        split="train",
        language="en",
        language_code="en-US",
        n_samples=12,
        seed=7,
    )
    repeated = sample_classification_rows(
        rows,
        split="train",
        language="en",
        language_code="en-US",
        n_samples=12,
        seed=7,
    )
    assert first.ids == repeated.ids
    assert {label: first.labels.count(label) for label in set(first.labels)} == {
        "0": 4,
        "1": 4,
        "2": 4,
    }


def test_layer_resolution_includes_best_raw_for_each_pooling() -> None:
    rows = [
        {
            "pooling": "mean",
            "condition": "raw",
            "metric": "r1",
            "layer": str(layer),
            "value": str(value),
        }
        for layer, value in ((0, 0.1), (1, 0.8), (2, 0.2))
    ]
    assert resolve_transfer_layers(
        rows,
        pooling="mean",
        selectors=("embedding", "best_raw", "intermediate", "final"),
    ) == (0, 1, 2)


def test_per_language_centering_recovers_shifted_zero_shot_classes() -> None:
    class_points = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    train_labels = tuple(str(index) for index in np.repeat(np.arange(3), 10))
    english = np.repeat(class_points, 10, axis=0)
    target = english + np.array([20.0, 20.0], dtype=np.float32)
    train = {
        "en": _bundle(english, language="en", split="train-en"),
        "ko": _bundle(target, language="ko", split="train-ko"),
    }
    evaluation = {
        "en": _bundle(english, language="en", split="test-en"),
        "ko": _bundle(target, language="ko", split="test-ko"),
    }
    rows = evaluate_crosslingual_transfer(
        train,
        evaluation,
        train_labels,
        {"en": train_labels, "ko": train_labels},
        layers=(24,),
    )
    accuracy = {
        str(row["condition"]): float(row["value"])
        for row in rows
        if row["metric"] == "accuracy" and row["test_language"] == "ko"
    }
    assert accuracy["per-language-centered"] == 1.0
    assert accuracy["per-language-centered"] > accuracy["raw"]
    assert accuracy["global-centered"] == accuracy["raw"]

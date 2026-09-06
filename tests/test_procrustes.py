from __future__ import annotations

import numpy as np

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.procrustes import (
    evaluate_procrustes,
    fit_orthogonal_procrustes,
)
from cross_lingual_representations.retrieval import evaluate_direction


def test_procrustes_recovers_rotated_space_and_orientation() -> None:
    rng = np.random.default_rng(42)
    train_source = rng.normal(size=(128, 12)).astype(np.float32)
    test_source = rng.normal(size=(32, 12)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    rotation = q.astype(np.float32)
    offset = rng.normal(size=(1, 12)).astype(np.float32)
    train_target = train_source @ rotation + offset
    test_target = test_source @ rotation + offset

    alignment = fit_orthogonal_procrustes(train_source, train_target)
    aligned_test = alignment.transform_source(test_source)
    centered_target = alignment.center_target(test_target)
    sentence_ids = tuple(map(str, range(len(test_source))))
    metrics = evaluate_direction(
        aligned_test,
        centered_target,
        sentence_ids,
        sentence_ids,
    )

    np.testing.assert_allclose(alignment.matrix.T @ alignment.matrix, np.eye(12), atol=1e-5)
    np.testing.assert_allclose(aligned_test, centered_target, atol=2e-5)
    assert metrics["r1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_degenerate_centered_layer_is_reported_as_nan() -> None:
    def make_bundle(split: str) -> RepresentationBundle:
        return RepresentationBundle(
            vectors=np.ones((2, 2, 1, 3), dtype=np.float32),
            model_name="test/model",
            model_revision=None,
            split=split,
            languages=("en", "ko"),
            language_codes=("eng_Latn", "kor_Hang"),
            sentence_ids=(f"{split}-1", f"{split}-2"),
            layer_labels=("layer_00_embedding",),
            pooling="last_token",
            seed=42,
            max_length=16,
            inference_dtype="float32",
            representation_dtype="float32",
            created_at="2026-01-01T00:00:00+00:00",
        )

    rows = evaluate_procrustes(make_bundle("dev"), make_bundle("devtest"))
    aligned = [row for row in rows if row["condition"] == "aligned"]
    centered = [row for row in rows if row["condition"] == "centered"]

    assert aligned
    assert centered
    assert all(row["status"] == "degenerate_zero_norm" for row in aligned)
    assert all(row["status"] == "degenerate_zero_norm" for row in centered)
    assert all(np.isnan(row["value"]) for row in aligned)
    assert all(np.isnan(row["value"]) for row in centered)

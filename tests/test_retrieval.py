from __future__ import annotations

import numpy as np

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.retrieval import (
    evaluate_all_directions,
    evaluate_direction,
    summarize_retrieval,
)


def test_identity_embeddings_retrieve_perfectly() -> None:
    identity = np.eye(6, dtype=np.float32)

    sentence_ids = tuple(map(str, range(6)))
    metrics = evaluate_direction(identity, identity, sentence_ids, sentence_ids)

    assert metrics["r1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_permuted_targets_use_sentence_ids() -> None:
    identity = np.eye(6, dtype=np.float32)
    permutation = np.array([2, 5, 1, 0, 4, 3])
    source_ids = tuple(map(str, range(6)))
    target_ids = tuple(str(index) for index in permutation)

    metrics = evaluate_direction(identity, identity[permutation], source_ids, target_ids)

    assert metrics["r1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_all_four_languages_produce_twelve_directions_per_layer() -> None:
    identity = np.eye(4, dtype=np.float32)
    vectors = np.stack([[identity, identity] for _ in range(4)]).transpose(0, 2, 1, 3)
    bundle = RepresentationBundle(
        vectors=vectors,
        model_name="test/model",
        model_revision=None,
        split="devtest",
        languages=("en", "ko", "ja", "zh"),
        language_codes=("eng_Latn", "kor_Hang", "jpn_Jpan", "zho_Hans"),
        sentence_ids=("0", "1", "2", "3"),
        layer_labels=("layer_00_embedding", "layer_01"),
        pooling="last_token",
        seed=42,
        max_length=16,
        inference_dtype="float32",
        representation_dtype="float32",
        created_at="2026-01-01T00:00:00+00:00",
    )

    rows = evaluate_all_directions(bundle)

    assert len(rows) == 2 * 12 * 4
    assert {row["source_language"] for row in rows} == {"en", "ko", "ja", "zh"}
    assert all(row["value"] == 1.0 for row in rows)
    summary = summarize_retrieval(rows)
    assert len(summary) == 2 * 4
    assert all(row["directions"] == 12 for row in summary)

from __future__ import annotations

import numpy as np

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.probes import evaluate_language_probe


def language_bundle(split: str, sentence_count: int, seed: int) -> RepresentationBundle:
    rng = np.random.default_rng(seed)
    language_count = 4
    states = 3
    hidden = 6
    vectors = np.empty((language_count, sentence_count, states, hidden), dtype=np.float32)
    for language in range(language_count):
        center = np.zeros(hidden, dtype=np.float32)
        center[language] = 10.0
        vectors[language] = center + rng.normal(
            scale=0.05, size=(sentence_count, states, hidden)
        )
    return RepresentationBundle(
        vectors=vectors,
        model_name="test/model",
        model_revision="revision",
        split=split,
        languages=("en", "ko", "ja", "zh"),
        language_codes=("eng_Latn", "kor_Hang", "jpn_Jpan", "zho_Hans"),
        sentence_ids=tuple(f"{split}-{index}" for index in range(sentence_count)),
        layer_labels=("layer_00_embedding", "layer_01", "layer_02"),
        pooling="last_token",
        seed=42,
        max_length=32,
        inference_dtype="float32",
        representation_dtype="float32",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_linear_probe_recovers_linearly_separable_languages() -> None:
    result = evaluate_language_probe(
        language_bundle("dev", 12, 1),
        language_bundle("devtest", 8, 2),
    )

    assert len(result.metrics) == 3 * 2
    assert all(row["value"] == 1.0 for row in result.metrics)
    assert len(result.confusion) == 3 * 4 * 4


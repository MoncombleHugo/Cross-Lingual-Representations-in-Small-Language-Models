from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cross_lingual_representations.cache import (
    CacheMismatchError,
    CacheSpec,
    RepresentationBundle,
    load_representation_cache,
    save_representation_cache,
)


def bundle() -> RepresentationBundle:
    return RepresentationBundle(
        vectors=np.arange(48, dtype=np.float16).reshape(2, 3, 2, 4),
        model_name="org/model",
        model_revision="abc123",
        split="devtest",
        languages=("en", "ko"),
        language_codes=("eng_Latn", "kor_Hang"),
        sentence_ids=("10", "20", "30"),
        layer_labels=("layer_00_embedding", "layer_01"),
        pooling="last_token",
        seed=42,
        max_length=128,
        inference_dtype="float32",
        representation_dtype="float16",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_cache_round_trip_without_pickle(tmp_path: Path) -> None:
    source = bundle()
    path = save_representation_cache(source, tmp_path / "cache.npz")

    restored = load_representation_cache(path, expected=source.spec)

    assert restored.metadata() == source.metadata()
    np.testing.assert_array_equal(restored.vectors, source.vectors)


def test_stale_cache_fails_clearly(tmp_path: Path) -> None:
    path = save_representation_cache(bundle(), tmp_path / "cache.npz")
    incompatible = CacheSpec(
        model_name="org/model",
        model_revision="abc123",
        split="devtest",
        languages=("en", "ko"),
        language_codes=("eng_Latn", "kor_Hang"),
        sentence_ids=("10", "20", "DIFFERENT"),
        pooling="last_token",
        max_length=128,
        seed=42,
        representation_dtype="float16",
    )

    with pytest.raises(CacheMismatchError, match="sentence_ids"):
        load_representation_cache(path, expected=incompatible)

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.centering import (
    centroid_sample_indices,
    evaluate_centering_sample_efficiency,
    sample_monolingual_rows,
)


def _bundle(
    vectors: np.ndarray[Any, Any], *, split: str, languages: tuple[str, ...]
) -> RepresentationBundle:
    return RepresentationBundle(
        vectors=vectors.astype(np.float32),
        model_name="model",
        model_revision=None,
        split=split,
        languages=languages,
        language_codes=languages,
        sentence_ids=tuple(str(index) for index in range(vectors.shape[1])),
        layer_labels=("embedding", "layer_01"),
        pooling="mean",
        seed=42,
        max_length=32,
        inference_dtype="float32",
        representation_dtype="float32",
        created_at="now",
    )


def test_monolingual_sampling_preserves_independent_ids() -> None:
    rows = [{"uid": f"x-{index}", "text": f"sentence {index}"} for index in range(5)]
    sample = sample_monolingual_rows(
        rows,
        split="corpus:train",
        language="en",
        language_code="en",
        n_samples=3,
        id_field="uid",
    )
    assert sample.ids == ("x-0", "x-1", "x-2")
    assert sample.sentences == {"en": ("sentence 0", "sentence 1", "sentence 2")}


def test_centroid_subsets_are_reproducible_and_validated() -> None:
    np.testing.assert_array_equal(
        centroid_sample_indices(20, 5, 7), centroid_sample_indices(20, 5, 7)
    )
    assert not np.array_equal(centroid_sample_indices(20, 5, 7), centroid_sample_indices(20, 5, 8))
    with pytest.raises(ValueError, match="no larger"):
        centroid_sample_indices(4, 5, 0)


def test_nonparallel_centering_recovers_offset_obscured_retrieval() -> None:
    # Semantic points are shared, but opposite large language offsets dominate raw cosine.
    semantics = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    offsets = {"en": np.array([0, 20], dtype=np.float32), "ko": np.array([20, 0], dtype=np.float32)}
    evaluation_values = np.stack(
        [np.stack([semantics + offsets[language]] * 2, axis=1) for language in ("en", "ko")]
    )
    train_values = evaluation_values[:, :3]
    evaluation = _bundle(evaluation_values, split="devtest", languages=("en", "ko"))
    train = _bundle(train_values, split="dev", languages=("en", "ko"))
    source = {
        language: _bundle(
            np.stack([semantics + offsets[language]] * 2, axis=1)[None, ...],
            split=f"monolingual:{language}",
            languages=(language,),
        )
        for language in ("en", "ko")
    }
    rows, selections = evaluate_centering_sample_efficiency(
        train, source, evaluation, sample_sizes=(4,), seeds=(0,)
    )
    final_r1 = {
        str(row["centering_condition"]): float(row["value"])
        for row in rows
        if row["layer"] == 1
        and row["metric"] == "r1"
        and row["source_language"] == "en"
        and row["target_language"] == "ko"
    }
    assert final_r1["nonparallel-centered"] == 1.0
    assert final_r1["nonparallel-centered"] > final_r1["raw"]
    assert selections["seed=0,n=4"]["en"] == ["0", "1", "2", "3"]


def test_centering_can_skip_repeated_controls() -> None:
    values = np.ones((2, 4, 2, 2), dtype=np.float32)
    evaluation = _bundle(values, split="devtest", languages=("en", "ko"))
    train = _bundle(values[:, :3], split="dev", languages=("en", "ko"))
    source = {
        language: _bundle(
            values[position : position + 1],
            split=f"monolingual:{language}",
            languages=(language,),
        )
        for position, language in enumerate(("en", "ko"))
    }
    rows, _ = evaluate_centering_sample_efficiency(
        train,
        source,
        evaluation,
        sample_sizes=(4,),
        seeds=(0,),
        include_controls=False,
    )
    assert {row["centering_condition"] for row in rows} == {
        "nonparallel-centered",
        "global-centered",
    }

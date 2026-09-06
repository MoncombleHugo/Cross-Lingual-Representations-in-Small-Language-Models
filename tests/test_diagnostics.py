from __future__ import annotations

import numpy as np
import pytest
import torch

from cross_lingual_representations.cache import RepresentationBundle
from cross_lingual_representations.diagnostics import (
    block_stage_retrieval_rows,
    bootstrap_retrieval_r1,
    geometry_diagnostics,
    position_retrieval_rows,
    query_subsample_peak_rows,
    relative_position_pool,
)
from cross_lingual_representations.similarity import l2_normalize


def _bundle() -> RepresentationBundle:
    identity = np.eye(8, dtype=np.float32)
    vectors = np.stack([[identity, identity] for _ in range(3)]).transpose(0, 2, 1, 3)
    return RepresentationBundle(
        vectors=vectors,
        model_name="test/model",
        model_revision=None,
        split="devtest",
        languages=("en", "ko", "ja"),
        language_codes=("eng_Latn", "kor_Hang", "jpn_Jpan"),
        sentence_ids=tuple(map(str, range(8))),
        layer_labels=("layer_00_embedding", "layer_01"),
        pooling="last_token",
        seed=42,
        max_length=16,
        inference_dtype="float32",
        representation_dtype="float32",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_geometry_diagnostics_report_expected_metrics() -> None:
    rows = geometry_diagnostics(_bundle())

    assert {row["metric"] for row in rows} == {
        "centroid_norm",
        "within_language_cosine",
        "stable_rank",
        "matched_cosine",
        "unmatched_cosine",
        "cosine_margin",
    }
    margins = [row["value"] for row in rows if row["metric"] == "cosine_margin"]
    assert margins and all(float(value) > 0.0 for value in margins)


def test_bootstrap_identity_retrieval_is_exact() -> None:
    rows = bootstrap_retrieval_r1(_bundle(), n_resamples=50)

    assert len(rows) == 2
    assert all(row["estimate"] == 1.0 for row in rows)
    assert all(row["lower"] == row["upper"] == 1.0 for row in rows)


def test_relative_position_retrieval_preserves_requested_layers() -> None:
    hidden = tuple(
        torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) + layer * 100
        for layer in range(3)
    )
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    pooled = relative_position_pool(
        hidden, mask, layers=(0, 2), fractions=(0.0, 0.5, 1.0)
    )

    assert pooled[0.0].shape == (2, 2, 3)
    np.testing.assert_array_equal(pooled[1.0][:, 0].numpy(), hidden[0][[0, 1], [2, 3]].numpy())
    identity = np.eye(8, dtype=np.float32)
    vectors = {0.0: np.stack([[identity, identity] for _ in range(3)]).transpose(0, 2, 1, 3)}
    rows = position_retrieval_rows(
        vectors,
        model_name="test/model",
        languages=("en", "ko", "ja"),
        sentence_ids=tuple(map(str, range(8))),
        layers=(5, 7),
    )
    assert {row["layer"] for row in rows} == {5, 7}
    assert all(row["value"] == 1.0 for row in rows)


def test_block_stage_retrieval_keeps_stage_labels() -> None:
    identity = np.eye(8, dtype=np.float32)
    values = np.stack([identity, identity, identity])
    vectors = {
        (7, "input", "last_token"): values,
        (7, "post_attention", "last_token"): values,
        (7, "output", "last_token"): values,
    }
    rows = block_stage_retrieval_rows(
        vectors,
        model_name="test/model",
        languages=("en", "ko", "ja"),
        sentence_ids=tuple(map(str, range(8))),
    )
    assert {row["stage"] for row in rows} == {"input", "post_attention", "output"}
    assert all(row["value"] == 1.0 for row in rows)


def test_non_finite_representations_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        l2_normalize(np.asarray([[1.0, np.inf]], dtype=np.float32))


def test_query_subsamples_recover_identity_peak() -> None:
    rows = query_subsample_peak_rows(_bundle(), n_subsamples=4, sample_size=5)

    assert len(rows) == 4
    assert all(row["best_layer"] == 0 for row in rows)
    assert all(row["best_r1"] == 1.0 for row in rows)

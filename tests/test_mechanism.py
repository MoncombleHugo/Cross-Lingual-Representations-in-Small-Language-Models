from __future__ import annotations

import numpy as np

from cross_lingual_representations.mechanism import evaluate_block_mechanism


def test_block_mechanism_reports_all_required_metrics() -> None:
    languages = ("en", "ko", "ja", "zh")
    semantic = np.eye(8, dtype=np.float32)
    offsets = np.zeros((4, 8), dtype=np.float32)
    offsets[:, :4] = 20.0 * np.eye(4, dtype=np.float32)
    values = semantic[None, :, :] + offsets[:, None, :]
    train = {(stage, "mean"): values for stage in ("input", "post_attention", "output")}
    evaluation = {key: item.copy() for key, item in train.items()}

    rows = evaluate_block_mechanism(
        train,
        evaluation,
        model_name="test/model",
        block_layer=7,
        languages=languages,
        train_ids=tuple(f"train-{index}" for index in range(8)),
        evaluation_ids=tuple(f"eval-{index}" for index in range(8)),
    )

    assert {row["stage"] for row in rows} == {"input", "post_attention", "output"}
    assert {
        "centroid_separation",
        "centroid_norm",
        "language_subspace_variance_fraction",
        "language_subspace_centroid_variance",
        "anisotropy",
        "stable_rank",
        "r1",
        "mrr",
        "language_probe_accuracy",
        "language_probe_macro_f1",
    } <= {row["metric"] for row in rows}
    centered = [
        float(row["value"])
        for row in rows
        if row["condition"] == "per-language-centered" and row["metric"] == "r1"
    ]
    assert centered and all(value == 1.0 for value in centered)

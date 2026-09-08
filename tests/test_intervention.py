from __future__ import annotations

import numpy as np

from cross_lingual_representations.intervention import evaluate_residual_interventions


def test_intervention_evaluation_keeps_conditions_and_layers() -> None:
    languages = ("en", "ko", "ja", "zh")
    identity = np.eye(8, dtype=np.float32)
    values = np.stack([[identity, identity] for _ in languages]).transpose(0, 2, 1, 3)
    train = {
        ("normal", None, "mean"): values,
        ("remove_language", None, "mean"): values,
    }
    evaluation = {key: item.copy() for key, item in train.items()}

    rows = evaluate_residual_interventions(
        train,
        evaluation,
        model_name="test/model",
        languages=languages,
        train_ids=tuple(f"train-{index}" for index in range(8)),
        evaluation_ids=tuple(f"eval-{index}" for index in range(8)),
        layer_indices=(7, 8),
        total_layers=8,
    )

    assert {row["condition"] for row in rows} == {"normal", "remove_language"}
    assert {row["layer"] for row in rows} == {7, 8}
    retrieval = [row for row in rows if row["metric"] == "r1"]
    assert retrieval and all(row["value"] == 1.0 for row in retrieval)

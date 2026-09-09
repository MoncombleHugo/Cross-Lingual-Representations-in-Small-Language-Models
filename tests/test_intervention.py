from __future__ import annotations

import numpy as np

from cross_lingual_representations.intervention import (
    evaluate_residual_interventions,
    intervention_bases,
    paired_retrieval_bootstrap,
    validate_intervention_basis,
)


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


def test_intervention_bases_are_train_derived_and_orthonormal() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(4, 20, 8)).astype(np.float32)
    values += np.arange(4, dtype=np.float32)[:, None, None] * np.array(
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    bases = intervention_bases(values, 3)
    assert set(bases) == {"language", "pca"}
    for basis in bases.values():
        validate_intervention_basis(basis)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=2e-5)


def test_paired_retrieval_bootstrap_reports_expected_zero_delta() -> None:
    languages = ("en", "ko", "ja", "zh")
    identity = np.eye(8, dtype=np.float32)
    values = np.stack([[identity] for _ in languages]).transpose(0, 2, 1, 3)
    evaluation = {
        ("normal", None, "mean"): values,
        ("remove_language", None, "mean"): values.copy(),
    }
    rows = paired_retrieval_bootstrap(
        evaluation,
        languages=languages,
        layer_indices=(24,),
        comparisons=(("remove_language", "normal"),),
        n_resamples=50,
    )
    assert len(rows) == 1
    assert rows[0]["value"] == 0.0
    assert rows[0]["ci_lower"] == 0.0
    assert rows[0]["ci_upper"] == 0.0

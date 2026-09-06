from dataclasses import replace
from pathlib import Path

import pytest

from cross_lingual_representations.config import load_experiment_config
from scripts import run_all
from scripts.run_all import build_steps, environment_report, output_roots


def test_smoke_outputs_are_isolated_and_pipeline_is_ordered() -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs" / "smoke.yaml"
    config = load_experiment_config(config_path)
    roots = output_roots(config, repository)
    steps = build_steps(config_path, config, roots, repository / "scripts")

    assert roots.cache == (repository / "artifacts" / "smoke" / "representations").resolve()
    assert roots.results == (repository / "results" / "smoke").resolve()
    assert [step.name for step in steps] == [
        "extract-train",
        "extract-evaluation",
        "retrieval",
        "diagnostics",
        "procrustes",
        "language-probe",
        "tokenization",
        "final-analysis",
    ]


def test_skip_extraction_starts_from_cached_analysis() -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs" / "smoke.yaml"
    config = load_experiment_config(config_path)
    roots = output_roots(config, repository)
    steps = build_steps(
        config_path,
        config,
        roots,
        repository / "scripts",
        skip_extraction=True,
    )

    assert steps[0].name == "retrieval"
    assert all(not step.name.startswith("extract-") for step in steps)


def test_official_flores_requires_local_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_experiment_config(repository / "configs" / "smoke.yaml")
    official = replace(config, dataset=replace(config.dataset, name="facebook/flores"))
    monkeypatch.setattr(run_all, "get_token", lambda: None)

    with pytest.raises(RuntimeError, match=r"gated.*hf.exe auth login"):
        environment_report(official)

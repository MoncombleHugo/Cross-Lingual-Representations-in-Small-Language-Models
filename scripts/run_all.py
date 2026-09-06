"""Run the complete cached experiment pipeline from one configuration file."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import get_token

from cross_lingual_representations.config import ExperimentConfig, load_experiment_config


@dataclass(frozen=True, slots=True)
class PipelineStep:
    """One independently executable pipeline stage."""

    name: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputRoots:
    """Locations shared by all pipeline stages."""

    cache: Path
    metadata: Path
    results: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--force", action="store_true", help="Re-extract representations")
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Require and reuse existing compatible representation caches",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stages without running them")
    return parser.parse_args()


def output_roots(
    config: ExperimentConfig,
    repository_root: Path,
    *,
    cache_root: Path | None = None,
    metadata_root: Path | None = None,
    results_root: Path | None = None,
) -> OutputRoots:
    """Resolve outputs, isolating smoke artifacts from full experiment caches."""
    smoke = config.experiment_name == "smoke"
    default_artifacts = repository_root / "artifacts" / ("smoke" if smoke else "")
    default_results = repository_root / "results" / ("smoke" if smoke else "")
    return OutputRoots(
        cache=(cache_root or default_artifacts / "representations").resolve(),
        metadata=(metadata_root or default_artifacts / "metadata").resolve(),
        results=(results_root or default_results).resolve(),
    )


def build_steps(
    config_path: Path,
    config: ExperimentConfig,
    roots: OutputRoots,
    script_root: Path,
    *,
    force: bool = False,
    skip_extraction: bool = False,
) -> list[PipelineStep]:
    """Build the ordered, restartable commands for an experiment."""
    python = sys.executable
    common_extraction = (
        "--config",
        str(config_path),
        "--cache-root",
        str(roots.cache),
        "--metadata-root",
        str(roots.metadata),
    )
    steps: list[PipelineStep] = []
    if not skip_extraction:
        for role in ("train", "evaluation"):
            command = [
                python,
                str(script_root / "extract_representations.py"),
                *common_extraction,
                "--split",
                role,
            ]
            if force:
                command.append("--force")
            steps.append(PipelineStep(f"extract-{role}", tuple(command)))

    common_analysis = (
        "--config",
        str(config_path),
        "--cache-root",
        str(roots.cache),
        "--results-root",
        str(roots.results),
    )
    steps.append(
        PipelineStep(
            "retrieval",
            (python, str(script_root / "run_retrieval.py"), *common_analysis),
        )
    )
    steps.append(
        PipelineStep(
            "diagnostics",
            (python, str(script_root / "run_diagnostics.py"), *common_analysis),
        )
    )
    if config.procrustes.enabled:
        steps.append(
            PipelineStep(
                "procrustes",
                (python, str(script_root / "run_procrustes.py"), *common_analysis),
            )
        )
    if config.language_probe.enabled:
        steps.append(
            PipelineStep(
                "language-probe",
                (python, str(script_root / "run_language_probe.py"), *common_analysis),
            )
        )
    if config.tokenization.enabled:
        steps.append(
            PipelineStep(
                "tokenization",
                (
                    python,
                    str(script_root / "run_tokenization_analysis.py"),
                    "--config",
                    str(config_path),
                    "--results-root",
                    str(roots.results),
                ),
            )
        )
    if config.procrustes.enabled and config.language_probe.enabled:
        steps.append(
            PipelineStep(
                "final-analysis",
                (
                    python,
                    str(script_root / "make_final_analysis.py"),
                    "--config",
                    str(config_path),
                    "--results-root",
                    str(roots.results),
                ),
            )
        )
    return steps


def environment_report(config: ExperimentConfig) -> dict[str, object]:
    """Inspect acceleration and local Hub authentication without exposing secrets."""
    cuda_available = torch.cuda.is_available()
    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "huggingface_authenticated": bool(get_token()),
    }
    if not cuda_available and shutil.which("nvidia-smi") is not None:
        report["acceleration_warning"] = (
            "An NVIDIA GPU is visible, but this Python environment has a CPU-only PyTorch build."
        )
    if config.dataset.name == "facebook/flores" and not report["huggingface_authenticated"]:
        raise RuntimeError(
            "facebook/flores is gated. Accept its access conditions on Hugging Face, then run "
            "`.venv\\Scripts\\hf.exe auth login` before using this configuration."
        )
    return report


def main() -> None:
    args = parse_args()
    if args.force and args.skip_extraction:
        raise ValueError("--force and --skip-extraction cannot be used together")
    repository_root = Path(__file__).resolve().parents[1]
    script_root = Path(__file__).resolve().parent
    config_path = args.config.resolve()
    config = load_experiment_config(config_path)
    roots = output_roots(
        config,
        repository_root,
        cache_root=args.cache_root,
        metadata_root=args.metadata_root,
        results_root=args.results_root,
    )
    runtime = environment_report(config)
    print(json.dumps(runtime, indent=2))
    if "acceleration_warning" in runtime:
        print(f"WARNING: {runtime['acceleration_warning']}")
    steps = build_steps(
        config_path,
        config,
        roots,
        script_root,
        force=args.force,
        skip_extraction=args.skip_extraction,
    )
    started_at = datetime.now(UTC)
    started = perf_counter()
    for index, step in enumerate(steps, start=1):
        printable = subprocess.list2cmdline(step.command)
        print(f"[{index}/{len(steps)}] {step.name}: {printable}", flush=True)
        if not args.dry_run:
            subprocess.run(step.command, cwd=repository_root, check=True)
    if args.dry_run:
        return

    elapsed = perf_counter() - started
    roots.metadata.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": config.experiment_name,
        "model": config.model.model_name,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": elapsed,
        "n_train_per_language": config.dataset.n_train,
        "n_eval_per_language": config.dataset.n_eval,
        "languages": list(config.dataset.languages),
        "cache_root": str(roots.cache),
        "results_root": str(roots.results),
        "steps": [step.name for step in steps],
        "environment": runtime,
    }
    manifest_path = roots.metadata / f"{config.experiment_name}_pipeline_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Completed {config.experiment_name} in {elapsed:.1f}s; "
        f"processed {config.dataset.n_train + config.dataset.n_eval} aligned sentence IDs "
        f"per language.\nCaches: {roots.cache}\nResults: {roots.results}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()

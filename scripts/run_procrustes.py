"""Fit Procrustes on dev caches and evaluate retrieval on devtest caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.experiment import load_configured_bundle
from cross_lingual_representations.plotting import plot_procrustes_r1
from cross_lingual_representations.procrustes import (
    evaluate_procrustes,
    summarize_procrustes,
)
from cross_lingual_representations.retrieval import read_tidy_csv, write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/representations"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if not config.procrustes.enabled:
        raise ValueError("Procrustes is disabled in this configuration")
    rows = []
    for pooling in config.pooling:
        train = load_configured_bundle(
            config,
            args.cache_root,
            role="train",
            pooling=pooling,
        )
        evaluation = load_configured_bundle(
            config,
            args.cache_root,
            role="evaluation",
            pooling=pooling,
        )
        rows.extend(evaluate_procrustes(train, evaluation))
    raw_path = args.results_root / "raw" / f"{config.experiment_name}_procrustes.csv"
    summary_path = args.results_root / "tables" / "procrustes_summary.csv"
    write_tidy_csv(rows, raw_path)
    write_tidy_csv(summarize_procrustes(rows), summary_path)
    saved_rows = read_tidy_csv(raw_path)
    for pooling in config.pooling:
        plot_procrustes_r1(
            saved_rows,
            args.results_root
            / "figures"
            / f"{config.experiment_name}_procrustes_{pooling}.png",
            pooling=pooling,
        )
    print(f"Saved Procrustes results: {raw_path}")
    print(f"Saved Procrustes summary: {summary_path}")


if __name__ == "__main__":
    main()

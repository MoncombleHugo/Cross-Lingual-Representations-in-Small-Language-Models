"""Evaluate centroid language-subspace dimensionality and removal controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.experiment import load_configured_cached_bundle
from cross_lingual_representations.language_subspace import evaluate_language_subspace
from cross_lingual_representations.plotting import (
    plot_language_subspace_variance,
    plot_subspace_removal_metric,
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
    settings = config.language_subspace
    if not settings.enabled:
        raise ValueError("Language-subspace analysis is disabled in this configuration")

    rows = []
    for pooling in config.pooling:
        print(f"Evaluating language subspace: {pooling}", flush=True)
        train = load_configured_cached_bundle(
            config, args.cache_root, role="train", pooling=pooling
        )
        evaluation = load_configured_cached_bundle(
            config, args.cache_root, role="evaluation", pooling=pooling
        )
        rows.extend(
            evaluate_language_subspace(
                train,
                evaluation,
                dimensions=settings.dimensions,
                random_seeds=settings.random_seeds,
                c=settings.c,
                max_iter=settings.max_iter,
                random_state=config.seed,
            )
        )

    raw_path = args.results_root / "raw" / f"{config.experiment_name}_language_subspace.csv"
    write_tidy_csv(rows, raw_path)
    saved_rows = read_tidy_csv(raw_path)
    figure_root = args.results_root / "figures"
    plot_language_subspace_variance(
        saved_rows,
        figure_root / f"{config.experiment_name}_language_subspace_variance.png",
    )
    plot_subspace_removal_metric(
        saved_rows,
        figure_root / f"{config.experiment_name}_subspace_removal_retrieval.png",
        record_type="retrieval",
        metric="r1",
        title="Translation retrieval after subspace removal",
        ylabel="Mean directed R@1",
    )
    plot_subspace_removal_metric(
        saved_rows,
        figure_root / f"{config.experiment_name}_subspace_removal_probe.png",
        record_type="language_probe",
        metric="accuracy",
        title="Language identity after subspace removal",
        ylabel="Probe accuracy",
    )
    print(f"Saved language-subspace results: {raw_path}")


if __name__ == "__main__":
    main()

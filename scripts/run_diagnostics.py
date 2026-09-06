"""Run cached geometry diagnostics and bootstrap retrieval intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.diagnostics import (
    bootstrap_retrieval_r1,
    geometry_diagnostics,
    query_subsample_peak_rows,
)
from cross_lingual_representations.experiment import load_configured_bundle
from cross_lingual_representations.plotting import (
    plot_bootstrap_retrieval,
    plot_geometry_diagnostics,
)
from cross_lingual_representations.retrieval import write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/representations"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--query-subsamples", type=int, default=20)
    parser.add_argument("--query-subsample-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    geometry_rows = []
    bootstrap_rows = []
    subsample_rows = []
    for pooling in config.pooling:
        evaluation = load_configured_bundle(
            config, args.cache_root, role="evaluation", pooling=pooling
        )
        geometry_rows.extend(geometry_diagnostics(evaluation, seed=config.seed))
        bootstrap_rows.extend(
            bootstrap_retrieval_r1(
                evaluation,
                n_resamples=args.bootstrap_resamples,
                seed=config.seed,
            )
        )
        subsample_rows.extend(
            query_subsample_peak_rows(
                evaluation,
                n_subsamples=args.query_subsamples,
                sample_size=min(args.query_subsample_size, len(evaluation.sentence_ids)),
                seed=config.seed,
            )
        )

    raw_root = args.results_root / "raw"
    geometry_path = raw_root / f"{config.experiment_name}_geometry.csv"
    bootstrap_path = raw_root / f"{config.experiment_name}_retrieval_bootstrap.csv"
    subsample_path = raw_root / f"{config.experiment_name}_retrieval_subsamples.csv"
    write_tidy_csv(geometry_rows, geometry_path)
    write_tidy_csv(bootstrap_rows, bootstrap_path)
    write_tidy_csv(subsample_rows, subsample_path)
    figure_root = args.results_root / "figures"
    plot_geometry_diagnostics(
        geometry_rows, figure_root / f"{config.experiment_name}_geometry_diagnostics.png"
    )
    plot_bootstrap_retrieval(
        bootstrap_rows, figure_root / f"{config.experiment_name}_retrieval_bootstrap.png"
    )
    print(f"Saved geometry diagnostics: {geometry_path}")
    print(f"Saved bootstrap intervals: {bootstrap_path}")
    print(f"Saved query-subsample stability: {subsample_path}")


if __name__ == "__main__":
    main()

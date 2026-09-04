"""Train layer-wise language probes on dev and evaluate on devtest."""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.experiment import load_configured_bundle
from cross_lingual_representations.plotting import (
    plot_alignment_vs_language,
    plot_language_probe_accuracy,
)
from cross_lingual_representations.probes import evaluate_language_probe
from cross_lingual_representations.retrieval import read_tidy_csv, write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/representations"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--retrieval-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if not config.language_probe.enabled:
        raise ValueError("Language probing is disabled in this configuration")
    metrics = []
    confusion = []
    for pooling in config.pooling:
        train = load_configured_bundle(config, args.cache_root, role="train", pooling=pooling)
        evaluation = load_configured_bundle(
            config, args.cache_root, role="evaluation", pooling=pooling
        )
        result = evaluate_language_probe(
            train,
            evaluation,
            c=config.language_probe.c,
            max_iter=config.language_probe.max_iter,
            random_state=config.seed,
        )
        metrics.extend(result.metrics)
        confusion.extend(result.confusion)
    raw_path = args.results_root / "raw" / f"{config.experiment_name}_language_probe.csv"
    confusion_path = (
        args.results_root / "raw" / f"{config.experiment_name}_probe_confusion.csv"
    )
    summary_path = args.results_root / "tables" / "language_probe_summary.csv"
    write_tidy_csv(metrics, raw_path)
    write_tidy_csv(confusion, confusion_path)
    write_tidy_csv(metrics, summary_path)
    plot_language_probe_accuracy(
        read_tidy_csv(raw_path),
        args.results_root / "figures" / f"{config.experiment_name}_language_probe.png",
    )
    retrieval_path = args.retrieval_csv or (
        args.results_root / "raw" / f"{config.experiment_name}_retrieval.csv"
    )
    if retrieval_path.exists():
        retrieval_rows = read_tidy_csv(retrieval_path)
        probe_rows = read_tidy_csv(raw_path)
        for pooling in config.pooling:
            plot_alignment_vs_language(
                retrieval_rows,
                probe_rows,
                args.results_root
                / "figures"
                / f"{config.experiment_name}_alignment_vs_language_{pooling}.png",
                pooling=pooling,
            )
    print(f"Saved language-probe metrics: {raw_path}")
    print(f"Saved selected-layer confusion matrices: {confusion_path}")


if __name__ == "__main__":
    main()

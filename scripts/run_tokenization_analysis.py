"""Compute tokenizer statistics for one or more model configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.experiment import load_configured_split
from cross_lingual_representations.models import load_tokenizer
from cross_lingual_representations.plotting import plot_tokenization_summary
from cross_lingual_representations.retrieval import read_tidy_csv, write_tidy_csv
from cross_lingual_representations.tokenization import (
    summarize_tokenization,
    tokenization_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for config_path in args.config:
        config = load_experiment_config(config_path)
        if not config.tokenization.enabled:
            continue
        dataset = load_configured_split(config, "evaluation")
        tokenizer = load_tokenizer(config.model)
        rows.extend(
            tokenization_rows(
                tokenizer,
                dataset,
                model_name=config.model.model_name,
                add_special_tokens=config.tokenization.add_special_tokens,
            )
        )
    if not rows:
        raise ValueError("Tokenization is disabled in every supplied configuration")
    raw_path = args.results_root / "raw" / "tokenization.csv"
    summary_path = args.results_root / "tables" / "tokenization_summary.csv"
    write_tidy_csv(rows, raw_path)
    write_tidy_csv(summarize_tokenization(rows), summary_path)
    saved_summary = read_tidy_csv(summary_path)
    plot_tokenization_summary(
        saved_summary,
        args.results_root / "figures" / "tokenization_summary.png",
    )
    print(f"Saved sentence-level tokenization data: {raw_path}")
    print(f"Saved tokenization summary: {summary_path}")


if __name__ == "__main__":
    main()

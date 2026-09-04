"""Consolidate saved model outputs into final tables and comparison figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_lingual_representations.analysis import summarize_models, summary_markdown
from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.plotting import (
    plot_model_retrieval_comparison,
    plot_retrieval_heatmap,
)
from cross_lingual_representations.procrustes import summarize_procrustes
from cross_lingual_representations.retrieval import (
    read_tidy_csv,
    summarize_retrieval,
    write_tidy_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [load_experiment_config(path) for path in args.config]
    retrieval_rows = []
    procrustes_rows = []
    probe_rows = []
    for config in configs:
        prefix = args.results_root / "raw" / config.experiment_name
        retrieval_rows.extend(read_tidy_csv(prefix.with_name(prefix.name + "_retrieval.csv")))
        procrustes_rows.extend(read_tidy_csv(prefix.with_name(prefix.name + "_procrustes.csv")))
        probe_rows.extend(read_tidy_csv(prefix.with_name(prefix.name + "_language_probe.csv")))

    tables = args.results_root / "tables"
    write_tidy_csv(summarize_retrieval(retrieval_rows), tables / "retrieval_summary.csv")
    write_tidy_csv(summarize_procrustes(procrustes_rows), tables / "procrustes_summary.csv")
    write_tidy_csv(probe_rows, tables / "language_probe_summary.csv")
    final_rows = summarize_models(retrieval_rows, procrustes_rows, probe_rows)
    write_tidy_csv(final_rows, tables / "final_summary.csv")
    (tables / "final_summary.md").write_text(
        summary_markdown(final_rows), encoding="utf-8"
    )

    figures = args.results_root / "figures"
    for pooling in {pooling for config in configs for pooling in config.pooling}:
        plot_model_retrieval_comparison(
            retrieval_rows,
            figures / f"model_retrieval_comparison_{pooling}.png",
            pooling=pooling,
        )
    for row in final_rows:
        experiment_name = next(
            config.experiment_name
            for config in configs
            if config.model.model_name == row["model"]
        )
        plot_retrieval_heatmap(
            retrieval_rows,
            figures / f"{experiment_name}_best_layer_heatmap.png",
            model=str(row["model"]),
            pooling=str(row["pooling"]),
            layer=int(row["best_layer"]),
        )
    print(f"Saved final summary: {tables / 'final_summary.md'}")


if __name__ == "__main__":
    main()

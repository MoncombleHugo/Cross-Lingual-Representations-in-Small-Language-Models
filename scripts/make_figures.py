"""Regenerate retrieval figures from a saved tidy CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_lingual_representations.plotting import plot_retrieval_r1
from cross_lingual_representations.retrieval import read_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_tidy_csv(args.retrieval_csv)
    poolings = sorted({row["pooling"] for row in rows})
    stem = args.retrieval_csv.stem
    for pooling in poolings:
        plot_retrieval_r1(
            rows,
            args.output_dir / f"{stem}_pairs_{pooling}.png",
            average_all_directions=False,
            pooling=pooling,
        )
    plot_retrieval_r1(
        rows,
        args.output_dir / f"{stem}_average.png",
        average_all_directions=True,
    )


if __name__ == "__main__":
    main()

"""Load a deterministic FLORES sample and print aligned examples."""

from __future__ import annotations

import argparse
import sys

from cross_lingual_representations.data import load_flores_split, preview_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "devtest"), default="dev")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    dataset = load_flores_split(split=args.split, n_samples=args.n_samples, seed=args.seed)
    print(preview_rows(dataset, count=args.preview))
    print(f"Loaded {len(dataset)} aligned rows from FLORES {dataset.split}.")


if __name__ == "__main__":
    main()

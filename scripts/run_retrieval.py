"""Evaluate all directed language pairs from cached representations."""

from __future__ import annotations

import argparse
from pathlib import Path

from cross_lingual_representations.cache import (
    CacheSpec,
    load_representation_cache,
    representation_cache_path,
)
from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.data import load_flores_split
from cross_lingual_representations.plotting import plot_retrieval_r1
from cross_lingual_representations.retrieval import (
    evaluate_all_directions,
    pair_similarity_rows,
    read_tidy_csv,
    summarize_retrieval,
    write_tidy_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/representations"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    split = config.dataset.evaluation_split
    dataset = load_flores_split(
        split=split,
        languages=config.dataset.languages,
        n_samples=config.dataset.n_eval,
        seed=config.seed,
        dataset_name=config.dataset.name,
        dataset_config=config.dataset.config,
    )
    retrieval_rows = []
    similarity_rows = []
    for pooling in config.pooling:
        path = representation_cache_path(args.cache_root, config.model.model_name, split, pooling)
        bundle = load_representation_cache(
            path,
            expected=CacheSpec(
                model_name=config.model.model_name,
                model_revision=config.model.revision,
                split=split,
                languages=tuple(dataset.sentences),
                language_codes=tuple(dataset.language_codes.values()),
                sentence_ids=dataset.ids,
                pooling=pooling,
                max_length=config.max_length,
                seed=config.seed,
                representation_dtype=config.representation_dtype,
            ),
        )
        retrieval_rows.extend(evaluate_all_directions(bundle))
        similarity_rows.extend(pair_similarity_rows(bundle, seed=config.seed))

    raw_path = args.results_root / "raw" / f"{config.experiment_name}_retrieval.csv"
    similarities_path = (
        args.results_root / "raw" / f"{config.experiment_name}_pair_similarities.csv"
    )
    write_tidy_csv(retrieval_rows, raw_path)
    write_tidy_csv(similarity_rows, similarities_path)
    summary_path = args.results_root / "tables" / "retrieval_summary.csv"
    write_tidy_csv(summarize_retrieval(retrieval_rows), summary_path)
    saved_retrieval_rows = read_tidy_csv(raw_path)
    figure_root = args.results_root / "figures"
    for pooling in config.pooling:
        plot_retrieval_r1(
            saved_retrieval_rows,
            figure_root / f"{config.experiment_name}_retrieval_pairs_{pooling}.png",
            average_all_directions=False,
            pooling=pooling,
        )
    plot_retrieval_r1(
        saved_retrieval_rows,
        figure_root / f"{config.experiment_name}_retrieval_average.png",
        average_all_directions=True,
    )
    print(f"Saved retrieval table: {raw_path}")
    print(f"Saved pair-similarity controls: {similarities_path}")
    print(f"Saved retrieval summary: {summary_path}")



if __name__ == "__main__":
    main()

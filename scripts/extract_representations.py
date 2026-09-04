"""Extract and cache pooled multilingual sentence representations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from cross_lingual_representations.cache import (
    CacheSpec,
    load_representation_cache,
    representation_cache_path,
    save_representation_cache,
)
from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.data import load_flores_split
from cross_lingual_representations.extraction import (
    RepresentationDType,
    extract_representations,
)
from cross_lingual_representations.models import load_causal_lm
from cross_lingual_representations.reproducibility import save_selected_ids, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "evaluation"), default="evaluation")
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/representations"))
    parser.add_argument("--metadata-root", type=Path, default=Path("artifacts/metadata"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    set_global_seed(config.seed)
    split = (
        config.dataset.train_split if args.split == "train" else config.dataset.evaluation_split
    )
    n_samples = config.dataset.n_train if args.split == "train" else config.dataset.n_eval
    dataset = load_flores_split(
        split=split,
        languages=config.dataset.languages,
        n_samples=n_samples,
        seed=config.seed,
        dataset_name=config.dataset.name,
        dataset_config=config.dataset.config,
    )
    save_selected_ids(
        args.metadata_root / f"{config.experiment_name}_{split}_indices.json",
        split=split,
        seed=config.seed,
        sentence_ids=dataset.ids,
    )

    missing = []
    for pooling in config.pooling:
        path = representation_cache_path(args.cache_root, config.model.model_name, split, pooling)
        expected = CacheSpec(
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
        )
        if path.exists() and not args.force:
            load_representation_cache(path, expected=expected)
            print(f"Reusing validated cache: {path}")
        else:
            missing.append(pooling)
    if not missing:
        return

    loaded = load_causal_lm(config.model)
    bundles = extract_representations(
        loaded,
        dataset,
        pooling_methods=missing,
        batch_size=config.batch_size,
        max_length=config.max_length,
        seed=config.seed,
        representation_dtype=cast(RepresentationDType, config.representation_dtype),
    )
    for pooling, bundle in bundles.items():
        path = representation_cache_path(args.cache_root, config.model.model_name, split, pooling)
        save_representation_cache(bundle, path)
        print(f"Saved {bundle.vectors.shape} {bundle.vectors.dtype} cache: {path}")


if __name__ == "__main__":
    main()

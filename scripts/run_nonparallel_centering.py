"""Estimate language centroids on unrelated text and evaluate FLORES retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from cross_lingual_representations.cache import (
    load_representation_cache,
    model_cache_name,
    representation_cache_path,
    save_representation_cache,
)
from cross_lingual_representations.centering import (
    evaluate_centering_sample_efficiency,
    load_monolingual_split,
    monolingual_cache_path,
)
from cross_lingual_representations.config import ExperimentConfig, load_experiment_config
from cross_lingual_representations.extraction import RepresentationDType, extract_representations
from cross_lingual_representations.models import load_causal_lm
from cross_lingual_representations.plotting import plot_centering_sample_efficiency
from cross_lingual_representations.reproducibility import set_global_seed
from cross_lingual_representations.retrieval import read_tidy_csv, write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/representations"))
    parser.add_argument("--metadata-root", type=Path, default=Path("artifacts/metadata"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--force-source", action="store_true")
    return parser.parse_args()


def load_existing_flores_bundle(
    config: ExperimentConfig, cache_root: Path, role: str, pooling: str
):
    """Reuse an existing cache without regenerating IDs from a mutable dataset mirror."""
    split = config.dataset.train_split if role == "train" else config.dataset.evaluation_split
    expected_count = config.dataset.n_train if role == "train" else config.dataset.n_eval
    path = representation_cache_path(cache_root, config.model.model_name, split, pooling)
    bundle = load_representation_cache(path)
    static_actual = (
        bundle.model_name,
        bundle.model_revision,
        bundle.split,
        bundle.languages,
        bundle.language_codes,
        bundle.pooling,
        bundle.max_length,
        bundle.seed,
        bundle.representation_dtype,
        len(bundle.sentence_ids),
    )
    static_expected = (
        config.model.model_name,
        config.model.revision,
        split,
        tuple(config.dataset.languages),
        tuple(config.dataset.languages.values()),
        pooling,
        config.max_length,
        config.seed,
        config.representation_dtype,
        expected_count,
    )
    if static_actual != static_expected:
        raise ValueError(f"Incompatible configured representation cache: {path}")
    return bundle


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    centering = config.nonparallel_centering
    if not centering.enabled or centering.dataset_configs is None:
        raise ValueError("nonparallel_centering must be enabled and configured")
    set_global_seed(config.seed)

    source_bundles_by_pooling = {pooling: {} for pooling in config.pooling}
    missing_by_language: dict[str, list[str]] = {}
    for language, dataset_config in centering.dataset_configs.items():
        for pooling in config.pooling:
            path = monolingual_cache_path(
                args.cache_root,
                model_cache_name(config.model.model_name),
                centering.dataset_name,
                language,
                pooling,
            )
            if path.exists() and not args.force_source:
                bundle = load_representation_cache(path)
                expected_split = (
                    f"monolingual:{centering.dataset_name}:{dataset_config}:{centering.split}"
                )
                static_actual = (
                    bundle.model_name,
                    bundle.model_revision,
                    bundle.split,
                    bundle.languages,
                    bundle.language_codes,
                    bundle.pooling,
                    bundle.max_length,
                    bundle.seed,
                    bundle.representation_dtype,
                    len(bundle.sentence_ids),
                )
                static_expected = (
                    config.model.model_name,
                    config.model.revision,
                    expected_split,
                    (language,),
                    (config.dataset.languages[language],),
                    pooling,
                    config.max_length,
                    config.seed,
                    config.representation_dtype,
                    centering.pool_size,
                )
                if static_actual != static_expected:
                    raise ValueError(f"Incompatible monolingual cache: {path}")
                source_bundles_by_pooling[pooling][language] = bundle
            else:
                missing_by_language.setdefault(language, []).append(pooling)

    if missing_by_language:
        missing_datasets = {}
        for language in missing_by_language:
            dataset_config = centering.dataset_configs[language]
            dataset = load_monolingual_split(
                dataset_name=centering.dataset_name,
                dataset_config=dataset_config,
                split=centering.split,
                language=language,
                language_code=config.dataset.languages[language],
                n_samples=centering.pool_size,
                seed=config.seed,
                shuffle_buffer_size=centering.shuffle_buffer_size,
                text_field=centering.text_field,
                id_field=centering.id_field,
                revision=centering.revision,
            )
            missing_datasets[language] = dataset
        loaded = load_causal_lm(config.model)
        for language, missing_pooling in missing_by_language.items():
            bundles = extract_representations(
                loaded,
                missing_datasets[language],
                pooling_methods=missing_pooling,
                batch_size=config.batch_size,
                max_length=config.max_length,
                seed=config.seed,
                representation_dtype=cast(RepresentationDType, config.representation_dtype),
            )
            for pooling, bundle in bundles.items():
                path = monolingual_cache_path(
                    args.cache_root,
                    model_cache_name(config.model.model_name),
                    centering.dataset_name,
                    language,
                    pooling,
                )
                save_representation_cache(bundle, path)
                source_bundles_by_pooling[pooling][language] = bundle

    all_rows = []
    all_selections = {}
    checkpoint_base = (
        args.results_root
        / "raw"
        / ".centering_checkpoints"
        / config.experiment_name
        / centering.dataset_name.replace("/", "__")
    )
    for pooling in config.pooling:
        train = load_existing_flores_bundle(config, args.cache_root, "train", pooling)
        evaluation = load_existing_flores_bundle(config, args.cache_root, "evaluation", pooling)
        checkpoint_root = checkpoint_base / pooling
        controls_path = checkpoint_root / "controls.csv"
        if not controls_path.exists():
            rows, _ = evaluate_centering_sample_efficiency(
                train,
                source_bundles_by_pooling[pooling],
                evaluation,
                sample_sizes=(),
                seeds=(),
            )
            write_tidy_csv(rows, controls_path)
            print(f"[{pooling}] saved controls checkpoint", flush=True)
        all_rows.extend(read_tidy_csv(controls_path))
        pooling_selections = {}
        for seed in centering.seeds:
            for n_samples in centering.sample_sizes:
                stem = f"seed_{seed:04d}_n_{n_samples:04d}"
                rows_path = checkpoint_root / f"{stem}.csv"
                ids_path = checkpoint_root / f"{stem}.json"
                if not rows_path.exists() or not ids_path.exists():
                    rows, selections = evaluate_centering_sample_efficiency(
                        train,
                        source_bundles_by_pooling[pooling],
                        evaluation,
                        sample_sizes=(n_samples,),
                        seeds=(seed,),
                        include_controls=False,
                    )
                    write_tidy_csv(rows, rows_path)
                    ids_path.write_text(
                        json.dumps(selections, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    print(
                        f"[{pooling}] saved seed={seed}, n={n_samples} checkpoint",
                        flush=True,
                    )
                all_rows.extend(read_tidy_csv(rows_path))
                pooling_selections.update(json.loads(ids_path.read_text(encoding="utf-8")))
        all_selections[pooling] = pooling_selections

    raw_path = (
        args.results_root / "raw" / f"{config.experiment_name}_centering_sample_efficiency.csv"
    )
    figure_path = (
        args.results_root
        / "figures"
        / f"{config.experiment_name}_centering_sample_efficiency.png"
    )
    metadata_path = (
        args.metadata_root / f"{config.experiment_name}_centering_sample_ids.json"
    )
    write_tidy_csv(all_rows, raw_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(all_selections, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_centering_sample_efficiency(read_tidy_csv(raw_path), figure_path)
    print(f"Saved centering results: {raw_path}")
    print(f"Saved centroid sample IDs: {metadata_path}")
    print(f"Saved centering figure: {figure_path}")


if __name__ == "__main__":
    main()

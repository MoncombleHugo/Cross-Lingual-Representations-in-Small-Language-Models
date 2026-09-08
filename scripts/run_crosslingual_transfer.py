"""Run English-only zero-shot MASSIVE intent transfer on frozen LM representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from cross_lingual_representations.cache import (
    CacheSpec,
    RepresentationBundle,
    load_representation_cache,
    model_cache_name,
    save_representation_cache,
)
from cross_lingual_representations.classification import (
    ClassificationSplit,
    evaluate_crosslingual_transfer,
    load_classification_split,
    resolve_transfer_layers,
    retrieval_transfer_gain_rows,
)
from cross_lingual_representations.config import ExperimentConfig, load_experiment_config
from cross_lingual_representations.extraction import RepresentationDType, extract_representations
from cross_lingual_representations.models import load_causal_lm, make_layer_labels
from cross_lingual_representations.plotting import (
    plot_crosslingual_transfer_by_layer,
    plot_retrieval_vs_transfer_gain,
)
from cross_lingual_representations.reproducibility import set_global_seed
from cross_lingual_representations.retrieval import read_tidy_csv, write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("artifacts/downstream_representations")
    )
    parser.add_argument("--metadata-root", type=Path, default=Path("artifacts/metadata"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--retrieval-csv", type=Path)
    parser.add_argument("--force-extraction", action="store_true")
    return parser.parse_args()


def classification_cache_path(
    root: Path,
    model_name: str,
    dataset_name: str,
    language: str,
    source_split: str,
    pooling: str,
) -> Path:
    return (
        root
        / model_cache_name(model_name)
        / dataset_name.replace("/", "__")
        / language
        / source_split
        / f"{pooling}.npz"
    )


def _load_or_extract(
    *,
    config: ExperimentConfig,
    datasets: dict[tuple[str, str], ClassificationSplit],
    selected_layers: tuple[int, ...],
    cache_root: Path,
    force: bool,
) -> dict[str, dict[str, dict[str, RepresentationBundle]]]:
    transfer = config.crosslingual_transfer
    expected_labels = tuple(
        make_layer_labels(max(selected_layers) + 1)[layer] for layer in selected_layers
    )
    loaded_bundles: dict[str, dict[str, dict[str, RepresentationBundle]]] = {
        pooling: {"train": {}, "evaluation": {}} for pooling in config.pooling
    }
    missing: dict[tuple[str, str], list[str]] = {}
    for (role, language), dataset in datasets.items():
        source_split = transfer.train_split if role == "train" else transfer.evaluation_split
        for pooling in config.pooling:
            path = classification_cache_path(
                cache_root,
                config.model.model_name,
                transfer.dataset_name,
                language,
                source_split,
                pooling,
            )
            expected = CacheSpec(
                model_name=config.model.model_name,
                model_revision=config.model.revision,
                split=dataset.split,
                languages=(language,),
                language_codes=(dataset.language_code,),
                sentence_ids=dataset.ids,
                pooling=pooling,
                max_length=transfer.max_length,
                seed=config.seed,
                representation_dtype=transfer.representation_dtype,
            )
            if path.exists() and not force:
                bundle = load_representation_cache(path, expected=expected)
                if bundle.layer_labels != expected_labels:
                    raise ValueError(f"Cached downstream layers do not match config: {path}")
                loaded_bundles[pooling][role][language] = bundle
            else:
                missing.setdefault((role, language), []).append(pooling)
    if missing:
        loaded_model = load_causal_lm(config.model)
        for (role, language), poolings in missing.items():
            dataset = datasets[(role, language)]
            bundles = extract_representations(
                loaded_model,
                dataset,
                pooling_methods=poolings,
                batch_size=transfer.batch_size,
                max_length=transfer.max_length,
                seed=config.seed,
                representation_dtype=cast(
                    RepresentationDType, transfer.representation_dtype
                ),
                layer_indices=selected_layers,
            )
            source_split = (
                transfer.train_split if role == "train" else transfer.evaluation_split
            )
            for pooling, bundle in bundles.items():
                path = classification_cache_path(
                    cache_root,
                    config.model.model_name,
                    transfer.dataset_name,
                    language,
                    source_split,
                    pooling,
                )
                save_representation_cache(bundle, path)
                loaded_bundles[pooling][role][language] = bundle
                print(f"Saved downstream cache: {path}", flush=True)
    return loaded_bundles


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    transfer = config.crosslingual_transfer
    if not transfer.enabled or transfer.dataset_configs is None:
        raise ValueError("crosslingual_transfer must be enabled and configured")
    set_global_seed(config.seed)
    retrieval_path = args.retrieval_csv or (
        args.results_root / "raw" / f"{config.experiment_name}_retrieval.csv"
    )
    retrieval_rows = read_tidy_csv(retrieval_path)
    layers_by_pooling = {
        pooling: resolve_transfer_layers(
            retrieval_rows, pooling=pooling, selectors=transfer.layers
        )
        for pooling in config.pooling
    }
    selected_layers = tuple(
        sorted({layer for layers in layers_by_pooling.values() for layer in layers})
    )
    datasets: dict[tuple[str, str], ClassificationSplit] = {}
    for language_index, (language, dataset_config) in enumerate(
        transfer.dataset_configs.items()
    ):
        for role, source_split, sample_count in (
            ("train", transfer.train_split, transfer.n_train),
            ("evaluation", transfer.evaluation_split, transfer.n_eval),
        ):
            datasets[(role, language)] = load_classification_split(
                dataset_name=transfer.dataset_name,
                dataset_config=dataset_config,
                source_split=source_split,
                language=language,
                language_code=config.dataset.languages[language],
                n_samples=sample_count,
                seed=config.seed + language_index,
                text_field=transfer.text_field,
                label_field=transfer.label_field,
                id_field=transfer.id_field,
                data_file_template=transfer.data_file_template,
            )
    bundles = _load_or_extract(
        config=config,
        datasets=datasets,
        selected_layers=selected_layers,
        cache_root=args.cache_root,
        force=args.force_extraction,
    )
    rows = []
    for pooling in config.pooling:
        rows.extend(
            evaluate_crosslingual_transfer(
                bundles[pooling]["train"],
                bundles[pooling]["evaluation"],
                datasets[("train", "en")].labels,
                {
                    language: datasets[("evaluation", language)].labels
                    for language in transfer.dataset_configs
                },
                layers=layers_by_pooling[pooling],
                c=transfer.c,
                max_iter=transfer.max_iter,
                random_state=config.seed,
            )
        )
    raw_path = (
        args.results_root / "raw" / f"{config.experiment_name}_crosslingual_transfer.csv"
    )
    figure_path = (
        args.results_root
        / "figures"
        / f"{config.experiment_name}_crosslingual_transfer_by_layer.png"
    )
    write_tidy_csv(rows, raw_path)
    plot_crosslingual_transfer_by_layer(read_tidy_csv(raw_path), figure_path)

    metadata_path = (
        args.metadata_root / f"{config.experiment_name}_crosslingual_transfer_samples.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": transfer.dataset_name,
                "layers_by_pooling": layers_by_pooling,
                "samples": {
                    f"{role}:{language}": list(dataset.ids)
                    for (role, language), dataset in datasets.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    all_gain_rows = []
    for transfer_csv in sorted((args.results_root / "raw").glob("*_crosslingual_transfer.csv")):
        experiment_name = transfer_csv.stem.removesuffix("_crosslingual_transfer")
        centering_csv = (
            args.results_root
            / "raw"
            / f"{experiment_name}_centering_sample_efficiency.csv"
        )
        if centering_csv.exists():
            all_gain_rows.extend(
                retrieval_transfer_gain_rows(
                    read_tidy_csv(transfer_csv), read_tidy_csv(centering_csv)
                )
            )
    gain_path = args.results_root / "raw" / "retrieval_vs_transfer_gain.csv"
    gain_figure_path = args.results_root / "figures" / "retrieval_vs_transfer_gain.png"
    write_tidy_csv(all_gain_rows, gain_path)
    plot_retrieval_vs_transfer_gain(read_tidy_csv(gain_path), gain_figure_path)
    print(f"Saved transfer results: {raw_path}")
    print(f"Saved transfer figure: {figure_path}")
    print(f"Saved retrieval-transfer gains: {gain_path}")
    print(f"Saved retrieval-transfer figure: {gain_figure_path}")
    print(f"Saved sample IDs: {metadata_path}")


if __name__ == "__main__":
    main()

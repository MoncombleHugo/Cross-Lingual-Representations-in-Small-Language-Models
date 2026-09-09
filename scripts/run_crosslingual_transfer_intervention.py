"""Evaluate FLORES-trained residual interventions on MASSIVE intent transfer."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from cross_lingual_representations.cache import (
    load_representation_cache,
    model_cache_name,
)
from cross_lingual_representations.classification import (
    ClassificationSplit,
    layer_number,
    load_classification_split,
)
from cross_lingual_representations.config import ExperimentConfig, load_experiment_config
from cross_lingual_representations.intervention import (
    intervention_bases,
    validate_intervention_basis,
)
from cross_lingual_representations.language_subspace import random_orthonormal_basis
from cross_lingual_representations.models import LoadedModel, load_causal_lm
from cross_lingual_representations.pooling import last_token_pool, mean_pool
from cross_lingual_representations.retrieval import read_tidy_csv, write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--block-cache-root", type=Path, default=Path("artifacts/block_representations")
    )
    parser.add_argument(
        "--normal-cache-root", type=Path, default=Path("artifacts/downstream_representations")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("artifacts/massive_intervention_representations")
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def _batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _decoder_layers(model: Any) -> Any:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("Expected a decoder model exposing model.layers")
    return layers


def _flores_bases(
    config: ExperimentConfig, root: Path
) -> tuple[dict[str, NDArray[np.float32]], int]:
    block_layer = config.block_mechanism.block_layer
    path = (
        root
        / model_cache_name(config.model.model_name)
        / f"block_{block_layer:02d}"
        / f"{config.dataset.train_split}.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("model_name") != config.model.model_name:
            raise ValueError("FLORES basis cache model mismatch")
        if metadata.get("split") != config.dataset.train_split:
            raise ValueError("FLORES intervention bases must be learned on dev")
        vectors = np.asarray(
            archive[f"output__{config.residual_intervention.basis_pooling}"],
            dtype=np.float32,
        )
    return intervention_bases(vectors, config.residual_intervention.dimensions), block_layer


def _load_datasets(config: ExperimentConfig) -> dict[tuple[str, str], ClassificationSplit]:
    transfer = config.crosslingual_transfer
    if transfer.dataset_configs is None:
        raise ValueError("MASSIVE dataset configs are required")
    datasets: dict[tuple[str, str], ClassificationSplit] = {}
    for language_index, (language, dataset_config) in enumerate(
        transfer.dataset_configs.items()
    ):
        roles = (("evaluation", transfer.evaluation_split, transfer.n_eval),)
        if language == "en":
            roles = (("train", transfer.train_split, transfer.n_train), *roles)
        for role, split, sample_count in roles:
            datasets[(role, language)] = load_classification_split(
                dataset_name=transfer.dataset_name,
                dataset_config=dataset_config,
                source_split=split,
                language=language,
                language_code=config.dataset.languages[language],
                n_samples=sample_count,
                seed=config.seed + language_index,
                text_field=transfer.text_field,
                label_field=transfer.label_field,
                id_field=transfer.id_field,
                data_file_template=transfer.data_file_template,
            )
    return datasets


def _normal_vectors(
    config: ExperimentConfig,
    datasets: Mapping[tuple[str, str], ClassificationSplit],
    root: Path,
) -> dict[tuple[str, str, str], NDArray[np.float32]]:
    transfer = config.crosslingual_transfer
    result: dict[tuple[str, str, str], NDArray[np.float32]] = {}
    for (role, language), dataset in datasets.items():
        source_split = transfer.train_split if role == "train" else transfer.evaluation_split
        for pooling in config.pooling:
            path = (
                root
                / model_cache_name(config.model.model_name)
                / transfer.dataset_name.replace("/", "__")
                / language
                / source_split
                / f"{pooling}.npz"
            )
            bundle = load_representation_cache(path)
            if bundle.sentence_ids != dataset.ids:
                raise ValueError(f"Normal MASSIVE cache IDs differ from sampled data: {path}")
            positions = [
                index
                for index, label in enumerate(bundle.layer_labels)
                if layer_number(label) == 24
            ]
            if len(positions) != 1:
                raise ValueError(f"Final hidden state is absent from {path}")
            result[(role, language, pooling)] = np.asarray(
                bundle.vectors[0, :, positions[0], :], dtype=np.float32
            )
    return result


def _cache_path(
    config: ExperimentConfig,
    root: Path,
    dataset: ClassificationSplit,
    role: str,
    condition: str,
    seed: int | None,
    block_layer: int,
) -> Path:
    suffix = condition if seed is None else f"{condition}_{seed}"
    return (
        root
        / model_cache_name(config.model.model_name)
        / config.crosslingual_transfer.dataset_name.replace("/", "__")
        / dataset.language
        / role
        / f"block_{block_layer:02d}"
        / f"{suffix}.npz"
    )


def _save_cache(
    path: Path,
    vectors: Mapping[str, NDArray[np.float32]],
    *,
    config: ExperimentConfig,
    dataset: ClassificationSplit,
    role: str,
    condition: str,
    seed: int | None,
    block_layer: int,
) -> None:
    metadata = {
        "model": config.model.model_name,
        "model_revision": config.model.revision,
        "dataset": config.crosslingual_transfer.dataset_name,
        "split": dataset.split,
        "role": role,
        "language": dataset.language,
        "sentence_ids": list(dataset.ids),
        "intervention_type": condition,
        "intervention_layer": block_layer,
        "basis_type": condition,
        "basis_dimension": config.residual_intervention.dimensions,
        "seed": seed,
        "experiment_seed": config.seed,
        "dtype": "float32",
        "pooling": list(config.pooling),
        "max_length": config.crosslingual_transfer.max_length,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **vectors, metadata=np.asarray(json.dumps(metadata)))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cache(
    path: Path,
    *,
    config: ExperimentConfig,
    dataset: ClassificationSplit,
    role: str,
    condition: str,
    seed: int | None,
    block_layer: int,
) -> dict[str, NDArray[np.float32]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        expected = {
            "model": config.model.model_name,
            "dataset": config.crosslingual_transfer.dataset_name,
            "split": dataset.split,
            "role": role,
            "language": dataset.language,
            "sentence_ids": list(dataset.ids),
            "intervention_type": condition,
            "intervention_layer": block_layer,
            "basis_type": condition,
            "basis_dimension": config.residual_intervention.dimensions,
            "seed": seed,
            "experiment_seed": config.seed,
            "dtype": "float32",
            "pooling": list(config.pooling),
            "max_length": config.crosslingual_transfer.max_length,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(f"Incompatible MASSIVE intervention cache: {mismatches}")
        return {
            pooling: np.asarray(archive[pooling], dtype=np.float32)
            for pooling in config.pooling
        }


def _extract_final(
    loaded: LoadedModel,
    dataset: ClassificationSplit,
    config: ExperimentConfig,
    basis: NDArray[np.float32],
    block_layer: int,
) -> dict[str, NDArray[np.float32]]:
    validate_intervention_basis(basis)
    block = _decoder_layers(loaded.model)[block_layer - 1]
    torch_basis = torch.as_tensor(basis, device=loaded.device, dtype=loaded.dtype)

    def intervene(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        cleaned = hidden - (hidden @ torch_basis) @ torch_basis.T
        return (cleaned, *output[1:]) if isinstance(output, tuple) else cleaned

    handle = block.register_forward_hook(intervene)
    poolers = {"last_token": last_token_pool, "mean": mean_pool}
    collected: dict[str, list[NDArray[np.float32]]] = {
        pooling: [] for pooling in config.pooling
    }
    try:
        for batch in tqdm(
            _batches(dataset.texts, config.crosslingual_transfer.batch_size),
            total=(len(dataset.texts) + config.crosslingual_transfer.batch_size - 1)
            // config.crosslingual_transfer.batch_size,
            desc=f"MASSIVE/{dataset.language}/{dataset.split}",
        ):
            encoded = loaded.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.crosslingual_transfer.max_length,
            )
            encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = loaded.model(
                    **encoded, output_hidden_states=True, use_cache=False, return_dict=True
                )
            final = outputs.hidden_states[-1]
            for pooling in config.pooling:
                values = poolers[pooling](final, encoded["attention_mask"])
                collected[pooling].append(
                    values.to(device="cpu", dtype=torch.float32).numpy()
                )
    finally:
        handle.remove()
    return {
        pooling: np.concatenate(values).astype(np.float32)
        for pooling, values in collected.items()
    }


def _evaluate(
    config: ExperimentConfig,
    datasets: Mapping[tuple[str, str], ClassificationSplit],
    vectors: Mapping[tuple[str, int | None, str, str, str], NDArray[np.float32]],
) -> list[dict[str, object]]:
    transfer = config.crosslingual_transfer
    conditions = sorted({(key[0], key[1]) for key in vectors})
    metric_rows: list[dict[str, object]] = []
    correctness: dict[tuple[str, int | None, str, str], NDArray[np.bool_]] = {}
    for condition, seed in conditions:
        for pooling in config.pooling:
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=transfer.c,
                    max_iter=transfer.max_iter,
                    random_state=config.seed,
                    solver="lbfgs",
                ),
            )
            classifier.fit(
                vectors[(condition, seed, pooling, "train", "en")],
                np.asarray(datasets[("train", "en")].labels),
            )
            for language in config.dataset.languages:
                truth = np.asarray(datasets[("evaluation", language)].labels)
                predictions = classifier.predict(
                    vectors[(condition, seed, pooling, "evaluation", language)]
                )
                correctness[(condition, seed, pooling, language)] = predictions == truth
                for metric, value in (
                    ("accuracy", accuracy_score(truth, predictions)),
                    ("macro_f1", f1_score(truth, predictions, average="macro")),
                ):
                    metric_rows.append(
                        {
                            "record_type": "metric",
                            "model": config.model.model_name,
                            "pooling": pooling,
                            "condition": condition,
                            "seed": "" if seed is None else seed,
                            "test_language": language,
                            "metric": metric,
                            "value": float(value),
                            "comparison": "",
                            "ci_lower": "",
                            "ci_upper": "",
                            "n_resamples": "",
                            "n_train": transfer.n_train,
                            "n_eval": transfer.n_eval,
                        }
                    )
    generator = np.random.default_rng(config.seed)
    samples = generator.integers(0, transfer.n_eval, size=(1000, transfer.n_eval))
    for pooling in config.pooling:
        for left, right in (("language", "normal"), ("pca", "normal"), ("language", "pca")):
            deltas = []
            observed = []
            for language in ("ko", "ja", "zh"):
                paired = (
                    correctness[(left, None, pooling, language)].astype(np.float64)
                    - correctness[(right, None, pooling, language)].astype(np.float64)
                )
                observed.append(float(paired.mean()))
                deltas.append(paired[samples].mean(axis=1))
            values = np.mean(np.stack(deltas), axis=0)
            metric_rows.append(
                {
                    "record_type": "bootstrap_delta",
                    "model": config.model.model_name,
                    "pooling": pooling,
                    "condition": left,
                    "seed": "",
                    "test_language": "target_mean",
                    "metric": "delta_accuracy",
                    "value": float(np.mean(observed)),
                    "comparison": f"{left}-{right}",
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "n_resamples": 1000,
                    "n_train": transfer.n_train,
                    "n_eval": transfer.n_eval * 3,
                }
            )
    return metric_rows


def _plot_summary(results_root: Path) -> None:
    rows = []
    for path in sorted((results_root / "raw").glob("*_massive_intervention.csv")):
        rows.extend(read_tidy_csv(path))
    rows = [
        row
        for row in rows
        if row["record_type"] == "metric"
        and row["metric"] == "accuracy"
        and row["test_language"] in {"ko", "ja", "zh"}
    ]
    if not rows:
        return
    models = sorted({str(row["model"]) for row in rows})
    figure = Figure(figsize=(10, 3.8 * len(models)), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(len(models), 2, squeeze=False)).reshape(len(models), 2)
    conditions = ("normal", "language", "pca", "random")
    for row_index, model in enumerate(models):
        for column, pooling in enumerate(("mean", "last_token")):
            axis = axes[row_index, column]
            values = []
            errors = []
            for condition in conditions:
                selected = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["pooling"] == pooling
                    and row["condition"] == condition
                ]
                if condition == "random":
                    by_seed: dict[str, list[float]] = {}
                    for row in selected:
                        by_seed.setdefault(str(row["seed"]), []).append(float(row["value"]))
                    seed_means = [float(np.mean(items)) for items in by_seed.values()]
                    values.append(float(np.mean(seed_means)))
                    errors.append(float(np.std(seed_means)))
                else:
                    values.append(float(np.mean([float(row["value"]) for row in selected])))
                    errors.append(0.0)
            axis.bar(conditions, values, yerr=errors, capsize=3)
            axis.set_ylim(0, 1)
            axis.set_ylabel("Mean KO/JA/ZH accuracy")
            axis.set_title(f"{model} — {pooling}")
            axis.grid(axis="y", alpha=0.25)
    destination = results_root / "figures" / "massive_intervention_summary.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    bases, block_layer = _flores_bases(config, args.block_cache_root)
    conditions: list[tuple[str, int | None, NDArray[np.float32] | None]] = [
        ("normal", None, None),
        ("language", None, bases["language"]),
        ("pca", None, bases["pca"]),
    ]
    for seed in config.residual_intervention.random_seeds:
        random = np.asarray(
            random_orthonormal_basis(
                bases["language"].shape[0], config.residual_intervention.dimensions, seed
            ),
            dtype=np.float32,
        )
        validate_intervention_basis(random, name=f"random seed {seed}")
        conditions.append(("random", seed, random))
    datasets = _load_datasets(config)
    normal = _normal_vectors(config, datasets, args.normal_cache_root)
    vectors: dict[tuple[str, int | None, str, str, str], NDArray[np.float32]] = {}
    for (role, language, pooling), values in normal.items():
        vectors[("normal", None, pooling, role, language)] = values

    pending = []
    for condition, seed, basis in conditions[1:]:
        assert basis is not None
        for (role, language), dataset in datasets.items():
            path = _cache_path(
                config, args.cache_root, dataset, role, condition, seed, block_layer
            )
            if path.exists():
                extracted = _load_cache(
                    path,
                    config=config,
                    dataset=dataset,
                    role=role,
                    condition=condition,
                    seed=seed,
                    block_layer=block_layer,
                )
                for pooling, values in extracted.items():
                    vectors[(condition, seed, pooling, role, language)] = values
            else:
                pending.append((condition, seed, basis, role, language, dataset, path))
    if pending:
        loaded = load_causal_lm(config.model)
        for condition, seed, basis, role, language, dataset, path in pending:
            print(f"Running MASSIVE {condition} seed={seed} {role}/{language}", flush=True)
            extracted = _extract_final(loaded, dataset, config, basis, block_layer)
            _save_cache(
                path,
                extracted,
                config=config,
                dataset=dataset,
                role=role,
                condition=condition,
                seed=seed,
                block_layer=block_layer,
            )
            for pooling, values in extracted.items():
                vectors[(condition, seed, pooling, role, language)] = values
    rows = _evaluate(config, datasets, vectors)
    output = args.results_root / "raw" / f"{config.experiment_name}_massive_intervention.csv"
    write_tidy_csv(rows, output)
    _plot_summary(args.results_root)
    print(f"Saved MASSIVE intervention results: {output}")


if __name__ == "__main__":
    main()

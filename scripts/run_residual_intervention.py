"""Remove a train-derived language subspace inside a block and continue the forward pass."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from cross_lingual_representations.cache import model_cache_name
from cross_lingual_representations.config import ExperimentConfig, load_experiment_config
from cross_lingual_representations.data import ParallelSplit
from cross_lingual_representations.experiment import (
    load_configured_cached_bundle,
    load_configured_split,
)
from cross_lingual_representations.intervention import (
    InterventionVectors,
    evaluate_residual_interventions,
)
from cross_lingual_representations.language_subspace import (
    centroid_language_subspace,
    random_orthonormal_basis,
)
from cross_lingual_representations.models import LoadedModel, load_causal_lm
from cross_lingual_representations.plotting import plot_residual_intervention
from cross_lingual_representations.pooling import last_token_pool, mean_pool
from cross_lingual_representations.retrieval import read_tidy_csv, write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/representations"))
    parser.add_argument(
        "--block-cache-root", type=Path, default=Path("artifacts/block_representations")
    )
    parser.add_argument(
        "--intervention-cache-root",
        type=Path,
        default=Path("artifacts/intervention_representations"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser.parse_args()


def _batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _decoder_layers(model: Any) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise TypeError("Expected a decoder model exposing model.layers")
    return layers


def _block_cache_path(config: ExperimentConfig, root: Path) -> Path:
    return (
        root
        / model_cache_name(config.model.model_name)
        / f"block_{config.block_mechanism.block_layer:02d}"
        / f"{config.dataset.train_split}.npz"
    )


def _language_basis(config: ExperimentConfig, root: Path) -> NDArray[np.float32]:
    path = _block_cache_path(config, root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing block-stage cache {path}; run scripts/run_block_mechanism.py first"
        )
    key = f"output__{config.residual_intervention.basis_pooling}"
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("model_name") != config.model.model_name:
            raise ValueError("Block-stage cache model does not match configuration")
        if metadata.get("block_layer") != config.block_mechanism.block_layer:
            raise ValueError("Block-stage cache layer does not match configuration")
        vectors = np.asarray(archive[key], dtype=np.float32)
    basis, _explained = centroid_language_subspace(vectors)
    return np.asarray(basis[:, : config.residual_intervention.dimensions], dtype=np.float32)


def _condition_cache_path(
    config: ExperimentConfig,
    root: Path,
    condition: str,
    seed: int | None,
    split: str,
) -> Path:
    suffix = condition if seed is None else f"{condition}_{seed}"
    return (
        root
        / model_cache_name(config.model.model_name)
        / f"block_{config.block_mechanism.block_layer:02d}"
        / suffix
        / f"{split}.npz"
    )


def _save_condition_cache(
    path: Path,
    vectors: dict[str, NDArray[np.float32]],
    *,
    config: ExperimentConfig,
    dataset: ParallelSplit,
    condition: str,
    seed: int | None,
    layer_indices: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": config.model.model_name,
        "model_revision": config.model.revision,
        "block_layer": config.block_mechanism.block_layer,
        "dimensions": config.residual_intervention.dimensions,
        "condition": condition,
        "seed": seed,
        "split": dataset.split,
        "languages": list(dataset.sentences),
        "language_codes": list(dataset.language_codes.values()),
        "sentence_ids": list(dataset.ids),
        "pooling": list(config.pooling),
        "layer_indices": list(layer_indices),
        "max_length": config.max_length,
        "experiment_seed": config.seed,
    }
    payload: dict[str, Any] = {**vectors, "metadata": np.asarray(json.dumps(metadata))}
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_condition_cache(
    path: Path,
    *,
    config: ExperimentConfig,
    role: str,
    condition: str,
    seed: int | None,
    layer_indices: tuple[int, ...],
) -> tuple[dict[str, NDArray[np.float32]], tuple[str, ...]]:
    split = config.dataset.train_split if role == "train" else config.dataset.evaluation_split
    expected_count = config.dataset.n_train if role == "train" else config.dataset.n_eval
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        expected = {
            "model_name": config.model.model_name,
            "model_revision": config.model.revision,
            "block_layer": config.block_mechanism.block_layer,
            "dimensions": config.residual_intervention.dimensions,
            "condition": condition,
            "seed": seed,
            "split": split,
            "languages": list(config.dataset.languages),
            "language_codes": list(config.dataset.languages.values()),
            "pooling": list(config.pooling),
            "layer_indices": list(layer_indices),
            "max_length": config.max_length,
            "experiment_seed": config.seed,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(f"Incompatible intervention cache fields: {mismatches}")
        ids = tuple(str(value) for value in metadata["sentence_ids"])
        if len(ids) != expected_count:
            raise ValueError("Intervention cache sample count does not match configuration")
        vectors = {
            pooling: np.asarray(archive[pooling], dtype=np.float32) for pooling in config.pooling
        }
    return vectors, ids


def _extract_intervention(
    loaded: LoadedModel,
    dataset: ParallelSplit,
    config: ExperimentConfig,
    basis: NDArray[np.float32],
    layer_indices: tuple[int, ...],
) -> dict[str, NDArray[np.float32]]:
    layers = _decoder_layers(loaded.model)
    block = layers[config.block_mechanism.block_layer - 1]
    torch_basis = torch.as_tensor(basis, device=loaded.device, dtype=loaded.dtype)

    def intervene(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        cleaned = hidden - (hidden @ torch_basis) @ torch_basis.T
        return (cleaned, *output[1:]) if isinstance(output, tuple) else cleaned

    handle = block.register_forward_hook(intervene)
    poolers = {"last_token": last_token_pool, "mean": mean_pool}
    collected: dict[str, list[NDArray[np.float32]]] = {pooling: [] for pooling in config.pooling}
    try:
        for language, texts in dataset.sentences.items():
            language_values: dict[str, list[NDArray[np.float32]]] = {
                pooling: [] for pooling in config.pooling
            }
            for text_batch in tqdm(
                _batches(texts, config.batch_size),
                total=(len(texts) + config.batch_size - 1) // config.batch_size,
                desc=f"intervene/{dataset.split}/{language}",
            ):
                encoded = loaded.tokenizer(
                    list(text_batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                )
                encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    outputs = loaded.model(
                        **encoded,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                states = outputs.hidden_states
                if states is None or max(layer_indices) >= len(states):
                    raise RuntimeError("Model returned incompatible hidden states")
                for pooling in config.pooling:
                    pooled = torch.stack(
                        [
                            poolers[pooling](states[layer], encoded["attention_mask"])
                            for layer in layer_indices
                        ],
                        dim=1,
                    )
                    language_values[pooling].append(
                        pooled.to(device="cpu", dtype=torch.float32).numpy()
                    )
            for pooling in config.pooling:
                collected[pooling].append(np.concatenate(language_values[pooling]))
    finally:
        handle.remove()
    return {
        pooling: np.stack(language_values).astype(np.float32)
        for pooling, language_values in collected.items()
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    settings = config.residual_intervention
    if not settings.enabled:
        raise ValueError("Residual intervention is disabled in this configuration")

    train_normal = {
        pooling: load_configured_cached_bundle(
            config, args.cache_root, role="train", pooling=pooling
        )
        for pooling in config.pooling
    }
    evaluation_normal = {
        pooling: load_configured_cached_bundle(
            config, args.cache_root, role="evaluation", pooling=pooling
        )
        for pooling in config.pooling
    }
    state_count = len(next(iter(train_normal.values())).layer_labels)
    layer_indices = tuple(range(config.block_mechanism.block_layer, state_count))
    train_ids = next(iter(train_normal.values())).sentence_ids
    evaluation_ids = next(iter(evaluation_normal.values())).sentence_ids
    train_vectors: InterventionVectors = {}
    evaluation_vectors: InterventionVectors = {}
    for pooling in config.pooling:
        train_vectors[("normal", None, pooling)] = np.asarray(
            train_normal[pooling].vectors[:, :, layer_indices, :], dtype=np.float32
        )
        evaluation_vectors[("normal", None, pooling)] = np.asarray(
            evaluation_normal[pooling].vectors[:, :, layer_indices, :], dtype=np.float32
        )

    language_basis = _language_basis(config, args.block_cache_root)
    conditions: list[tuple[str, int | None, NDArray[np.float32]]] = [
        ("remove_language", None, language_basis)
    ]
    for seed in settings.random_seeds:
        random_basis = random_orthonormal_basis(language_basis.shape[0], settings.dimensions, seed)
        conditions.append(("remove_random", seed, np.asarray(random_basis, dtype=np.float32)))

    pending: list[tuple[str, int | None, NDArray[np.float32], str]] = []
    for condition, seed, basis in conditions:
        for role in ("train", "evaluation"):
            split = (
                config.dataset.train_split if role == "train" else config.dataset.evaluation_split
            )
            path = _condition_cache_path(
                config, args.intervention_cache_root, condition, seed, split
            )
            if path.exists():
                vectors, ids = _load_condition_cache(
                    path,
                    config=config,
                    role=role,
                    condition=condition,
                    seed=seed,
                    layer_indices=layer_indices,
                )
                destination = train_vectors if role == "train" else evaluation_vectors
                for pooling, values in vectors.items():
                    destination[(condition, seed, pooling)] = values
                if role == "train" and ids != train_ids:
                    raise ValueError("Intervention train IDs differ from normal cache")
                if role == "evaluation" and ids != evaluation_ids:
                    raise ValueError("Intervention evaluation IDs differ from normal cache")
            else:
                pending.append((condition, seed, basis, role))

    if pending:
        datasets = {
            role: load_configured_split(config, role)
            for role in {pending_role for *_prefix, pending_role in pending}
        }
        loaded = load_causal_lm(config.model)
        for condition, seed, basis, role in pending:
            dataset = datasets[role]
            print(f"Running {condition} seed={seed} on {dataset.split}", flush=True)
            values = _extract_intervention(loaded, dataset, config, basis, layer_indices)
            _save_condition_cache(
                _condition_cache_path(
                    config,
                    args.intervention_cache_root,
                    condition,
                    seed,
                    dataset.split,
                ),
                values,
                config=config,
                dataset=dataset,
                condition=condition,
                seed=seed,
                layer_indices=layer_indices,
            )
            destination = train_vectors if role == "train" else evaluation_vectors
            for pooling, vectors in values.items():
                destination[(condition, seed, pooling)] = vectors

    rows = evaluate_residual_interventions(
        train_vectors,
        evaluation_vectors,
        model_name=config.model.model_name,
        languages=tuple(config.dataset.languages),
        train_ids=train_ids,
        evaluation_ids=evaluation_ids,
        layer_indices=layer_indices,
        total_layers=state_count - 1,
        c=settings.c,
        max_iter=settings.max_iter,
        random_state=config.seed,
    )
    raw_path = (
        args.results_root
        / "raw"
        / (
            f"{config.experiment_name}_block_"
            f"{config.block_mechanism.block_layer:02d}_intervention.csv"
        )
    )
    figure_path = (
        args.results_root
        / "figures"
        / (
            f"{config.experiment_name}_block_"
            f"{config.block_mechanism.block_layer:02d}_intervention.png"
        )
    )
    write_tidy_csv(rows, raw_path)
    plot_residual_intervention(read_tidy_csv(raw_path), figure_path)
    print(f"Saved residual intervention: {raw_path}")


if __name__ == "__main__":
    main()

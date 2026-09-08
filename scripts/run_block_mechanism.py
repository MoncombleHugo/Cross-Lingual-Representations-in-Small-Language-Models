"""Locate language-specific geometry inside one configured decoder block."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from cross_lingual_representations.cache import model_cache_name
from cross_lingual_representations.config import ExperimentConfig, load_experiment_config
from cross_lingual_representations.data import ParallelSplit
from cross_lingual_representations.experiment import load_configured_split
from cross_lingual_representations.mechanism import StageVectors, evaluate_block_mechanism
from cross_lingual_representations.models import LoadedModel, load_causal_lm
from cross_lingual_representations.plotting import plot_block_mechanism
from cross_lingual_representations.pooling import last_token_pool, mean_pool
from cross_lingual_representations.retrieval import read_tidy_csv, write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/block_representations")
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--block-layer", type=int, help="Override the configured diagnostic block")
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


def _cache_path(config: ExperimentConfig, root: Path, split: str) -> Path:
    block = config.block_mechanism.block_layer
    return root / model_cache_name(config.model.model_name) / f"block_{block:02d}" / f"{split}.npz"


def _save_stage_cache(
    path: Path,
    vectors: StageVectors,
    *,
    config: ExperimentConfig,
    dataset: ParallelSplit,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": config.model.model_name,
        "model_revision": config.model.revision,
        "block_layer": config.block_mechanism.block_layer,
        "split": dataset.split,
        "languages": list(dataset.sentences),
        "language_codes": list(dataset.language_codes.values()),
        "sentence_ids": list(dataset.ids),
        "pooling": list(config.pooling),
        "max_length": config.max_length,
        "seed": config.seed,
    }
    payload: dict[str, Any] = {
        f"{stage}__{pooling}": values for (stage, pooling), values in vectors.items()
    }
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_stage_cache(
    path: Path, *, config: ExperimentConfig, role: str
) -> tuple[dict[tuple[str, str], NDArray[np.float32]], tuple[str, ...]]:
    split = config.dataset.train_split if role == "train" else config.dataset.evaluation_split
    expected_count = config.dataset.n_train if role == "train" else config.dataset.n_eval
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        expected = {
            "model_name": config.model.model_name,
            "model_revision": config.model.revision,
            "block_layer": config.block_mechanism.block_layer,
            "split": split,
            "languages": list(config.dataset.languages),
            "language_codes": list(config.dataset.languages.values()),
            "pooling": list(config.pooling),
            "max_length": config.max_length,
            "seed": config.seed,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(f"Incompatible block cache fields: {mismatches}")
        sentence_ids = tuple(str(value) for value in metadata["sentence_ids"])
        if len(sentence_ids) != expected_count:
            raise ValueError("Block cache sample count does not match configuration")
        vectors = {
            (stage, pooling): np.asarray(archive[f"{stage}__{pooling}"], dtype=np.float32)
            for stage in ("input", "post_attention", "output")
            for pooling in config.pooling
        }
    return vectors, sentence_ids


def _extract_stage_vectors(
    loaded: LoadedModel,
    dataset: ParallelSplit,
    config: ExperimentConfig,
) -> dict[tuple[str, str], NDArray[np.float32]]:
    layers = _decoder_layers(loaded.model)
    block_index = config.block_mechanism.block_layer - 1
    if block_index < 0 or block_index >= len(layers):
        raise ValueError(f"Configured block layer must be between 1 and {len(layers)}")
    block = layers[block_index]
    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module: Any, inputs: tuple[Any, ...]) -> None:
        captured["input"] = inputs[0]

    def capture_attention(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        captured["attention"] = output[0] if isinstance(output, tuple) else output

    def capture_output(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        captured["output"] = output[0] if isinstance(output, tuple) else output

    handles = (
        block.register_forward_pre_hook(capture_input),
        block.self_attn.register_forward_hook(capture_attention),
        block.register_forward_hook(capture_output),
    )
    poolers = {"last_token": last_token_pool, "mean": mean_pool}
    collected: dict[tuple[str, str], list[NDArray[np.float32]]] = {
        (stage, pooling): []
        for stage in ("input", "post_attention", "output")
        for pooling in config.pooling
    }
    by_language: dict[tuple[str, str], list[NDArray[np.float32]]]
    try:
        for language, texts in dataset.sentences.items():
            by_language = {key: [] for key in collected}
            for text_batch in tqdm(
                _batches(texts, config.batch_size),
                total=(len(texts) + config.batch_size - 1) // config.batch_size,
                desc=f"block-{config.block_mechanism.block_layer}/{dataset.split}/{language}",
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
                    loaded.model(**encoded, use_cache=False, return_dict=True)
                stage_values = {
                    "input": captured["input"],
                    "post_attention": captured["input"] + captured["attention"],
                    "output": captured["output"],
                }
                for stage, values in stage_values.items():
                    for pooling in config.pooling:
                        pooled = poolers[pooling](values, encoded["attention_mask"])
                        by_language[(stage, pooling)].append(
                            pooled.to(device="cpu", dtype=torch.float32).numpy()
                        )
            for key in collected:
                collected[key].append(np.concatenate(by_language[key]))
    finally:
        for handle in handles:
            handle.remove()
    return {
        key: np.stack(language_values).astype(np.float32)
        for key, language_values in collected.items()
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.block_layer is not None:
        config = replace(
            config,
            block_mechanism=replace(config.block_mechanism, block_layer=args.block_layer),
        )
    settings = config.block_mechanism
    if not settings.enabled:
        raise ValueError("Block-mechanism analysis is disabled in this configuration")

    cached: dict[str, tuple[dict[tuple[str, str], NDArray[np.float32]], tuple[str, ...]]] = {}
    missing_roles = []
    for role in ("train", "evaluation"):
        split = config.dataset.train_split if role == "train" else config.dataset.evaluation_split
        path = _cache_path(config, args.artifact_root, split)
        if path.exists():
            cached[role] = _load_stage_cache(path, config=config, role=role)
        else:
            missing_roles.append(role)

    if missing_roles:
        datasets = {role: load_configured_split(config, role) for role in missing_roles}
        loaded = load_causal_lm(config.model)
        for role, dataset in datasets.items():
            print(f"Extracting block stages for {dataset.split}", flush=True)
            vectors = _extract_stage_vectors(loaded, dataset, config)
            _save_stage_cache(
                _cache_path(config, args.artifact_root, dataset.split),
                vectors,
                config=config,
                dataset=dataset,
            )
            cached[role] = (vectors, dataset.ids)

    train_vectors, train_ids = cached["train"]
    evaluation_vectors, evaluation_ids = cached["evaluation"]
    rows = evaluate_block_mechanism(
        train_vectors,
        evaluation_vectors,
        model_name=config.model.model_name,
        block_layer=settings.block_layer,
        languages=tuple(config.dataset.languages),
        train_ids=train_ids,
        evaluation_ids=evaluation_ids,
        subspace_dimensions=settings.subspace_dimensions,
        c=settings.c,
        max_iter=settings.max_iter,
        random_state=config.seed,
    )
    raw_path = (
        args.results_root
        / "raw"
        / f"{config.experiment_name}_block_{settings.block_layer:02d}_mechanism.csv"
    )
    figure_path = (
        args.results_root
        / "figures"
        / f"{config.experiment_name}_block_{settings.block_layer:02d}_mechanism.png"
    )
    write_tidy_csv(rows, raw_path)
    plot_block_mechanism(read_tidy_csv(raw_path), figure_path)
    print(f"Saved block-mechanism analysis: {raw_path}")


if __name__ == "__main__":
    main()

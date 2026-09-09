"""Measure next-token NLL under FLORES-trained residual interventions."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import NDArray
from tqdm.auto import tqdm

from cross_lingual_representations.cache import model_cache_name
from cross_lingual_representations.config import ExperimentConfig, load_experiment_config
from cross_lingual_representations.data import ParallelSplit
from cross_lingual_representations.experiment import load_configured_split
from cross_lingual_representations.intervention import intervention_bases
from cross_lingual_representations.models import LoadedModel, load_causal_lm
from cross_lingual_representations.retrieval import write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--block-cache-root", type=Path, default=Path("artifacts/block_representations")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("artifacts/intervention_lm_eval")
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
        if metadata.get("split") != "dev":
            raise ValueError("Language and PCA intervention bases must be learned on FLORES dev")
        vectors = np.asarray(
            archive[f"output__{config.residual_intervention.basis_pooling}"],
            dtype=np.float32,
        )
    return intervention_bases(vectors, config.residual_intervention.dimensions), block_layer


def _cache_path(
    config: ExperimentConfig, root: Path, condition: str, language: str
) -> Path:
    return (
        root
        / model_cache_name(config.model.model_name)
        / f"block_{config.block_mechanism.block_layer:02d}"
        / condition
        / f"{language}.npz"
    )


def _save_cache(
    path: Path,
    sentence_nll_sum: NDArray[np.float64],
    token_count: NDArray[np.int64],
    *,
    config: ExperimentConfig,
    dataset: ParallelSplit,
    condition: str,
    language: str,
) -> None:
    metadata = {
        "model": config.model.model_name,
        "model_revision": config.model.revision,
        "dataset": config.dataset.name,
        "split": dataset.split,
        "language": language,
        "sentence_ids": list(dataset.ids),
        "intervention_type": condition,
        "intervention_layer": config.block_mechanism.block_layer,
        "basis_type": condition,
        "basis_dimension": 0 if condition == "normal" else config.residual_intervention.dimensions,
        "seed": config.seed,
        "dtype": "float64",
        "max_length": config.max_length,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                sentence_nll_sum=sentence_nll_sum,
                token_count=token_count,
                metadata=np.asarray(json.dumps(metadata)),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cache(
    path: Path,
    *,
    config: ExperimentConfig,
    dataset: ParallelSplit,
    condition: str,
    language: str,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        expected = {
            "model": config.model.model_name,
            "dataset": config.dataset.name,
            "split": dataset.split,
            "language": language,
            "sentence_ids": list(dataset.ids),
            "intervention_type": condition,
            "intervention_layer": config.block_mechanism.block_layer,
            "basis_type": condition,
            "basis_dimension": (
                0 if condition == "normal" else config.residual_intervention.dimensions
            ),
            "seed": config.seed,
            "dtype": "float64",
            "max_length": config.max_length,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(f"Incompatible LM-evaluation cache: {mismatches}")
        return (
            np.asarray(archive["sentence_nll_sum"], dtype=np.float64),
            np.asarray(archive["token_count"], dtype=np.int64),
        )


def _evaluate_language(
    loaded: LoadedModel,
    texts: Sequence[str],
    config: ExperimentConfig,
    basis: NDArray[np.float32] | None,
    condition: str,
    language: str,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    handle = None
    if basis is not None:
        block = _decoder_layers(loaded.model)[config.block_mechanism.block_layer - 1]
        torch_basis = torch.as_tensor(basis, device=loaded.device, dtype=loaded.dtype)

        def intervene(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            cleaned = hidden - (hidden @ torch_basis) @ torch_basis.T
            return (cleaned, *output[1:]) if isinstance(output, tuple) else cleaned

        handle = block.register_forward_hook(intervene)
    sums: list[NDArray[np.float64]] = []
    counts: list[NDArray[np.int64]] = []
    try:
        for batch in tqdm(
            _batches(texts, config.batch_size),
            total=(len(texts) + config.batch_size - 1) // config.batch_size,
            desc=f"LM/{condition}/{language}",
        ):
            encoded = loaded.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )
            encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = loaded.model(**encoded, use_cache=False, return_dict=True).logits
            targets = encoded["input_ids"][:, 1:]
            valid = encoded["attention_mask"][:, 1:].to(torch.bool)
            losses = functional.cross_entropy(
                logits[:, :-1, :].float().transpose(1, 2), targets, reduction="none"
            )
            sums.append(
                (losses * valid).sum(dim=1).to(device="cpu", dtype=torch.float64).numpy()
            )
            counts.append(valid.sum(dim=1).to(device="cpu", dtype=torch.int64).numpy())
    finally:
        if handle is not None:
            handle.remove()
    result_sums = np.concatenate(sums)
    result_counts = np.concatenate(counts)
    if np.any(result_counts <= 0) or not np.isfinite(result_sums).all():
        raise ValueError("LM evaluation produced invalid token counts or NLL values")
    return result_sums, result_counts


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    bases, _block_layer = _flores_bases(config, args.block_cache_root)
    dataset = load_configured_split(config, "evaluation")
    if dataset.split != "devtest":
        raise ValueError("LM evaluation must use FLORES devtest")
    conditions: tuple[tuple[str, NDArray[np.float32] | None], ...] = (
        ("normal", None),
        ("language", bases["language"]),
        ("pca", bases["pca"]),
    )
    cached: dict[tuple[str, str], tuple[NDArray[np.float64], NDArray[np.int64]]] = {}
    pending = []
    for condition, basis in conditions:
        for language, texts in dataset.sentences.items():
            path = _cache_path(config, args.cache_root, condition, language)
            if path.exists():
                cached[(condition, language)] = _load_cache(
                    path,
                    config=config,
                    dataset=dataset,
                    condition=condition,
                    language=language,
                )
            else:
                pending.append((condition, basis, language, texts, path))
    if pending:
        loaded = load_causal_lm(config.model)
        for condition, basis, language, texts, path in pending:
            values = _evaluate_language(
                loaded, texts, config, basis, condition, language
            )
            _save_cache(
                path,
                *values,
                config=config,
                dataset=dataset,
                condition=condition,
                language=language,
            )
            cached[(condition, language)] = values
    nll = {
        key: float(sentence_sums.sum() / token_counts.sum())
        for key, (sentence_sums, token_counts) in cached.items()
    }
    rows = []
    for condition, _basis in conditions:
        for language in dataset.sentences:
            value = nll[(condition, language)]
            rows.append(
                {
                    "model": config.model.model_name,
                    "language": language,
                    "condition": condition,
                    "metric": "mean_token_nll",
                    "value": value,
                    "perplexity": math.exp(value) if value < 700 else float("inf"),
                    "delta_nll": value - nll[("normal", language)],
                    "sentences": len(dataset.ids),
                    "valid_tokens": int(cached[(condition, language)][1].sum()),
                    "basis_split": "dev" if condition != "normal" else "",
                    "evaluation_split": "devtest",
                    "intervention_layer": config.block_mechanism.block_layer,
                    "basis_dimension": (
                        0 if condition == "normal" else config.residual_intervention.dimensions
                    ),
                }
            )
    output = args.results_root / "raw" / f"{config.experiment_name}_intervention_lm_eval.csv"
    write_tidy_csv(rows, output)
    print(f"Saved intervention LM evaluation: {output}")


if __name__ == "__main__":
    main()

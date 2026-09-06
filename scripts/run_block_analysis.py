"""Decompose selected decoder blocks into input, post-attention, and output retrieval."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from cross_lingual_representations.config import load_experiment_config
from cross_lingual_representations.data import load_flores_split
from cross_lingual_representations.diagnostics import block_stage_retrieval_rows
from cross_lingual_representations.models import load_causal_lm
from cross_lingual_representations.plotting import plot_block_stage_retrieval
from cross_lingual_representations.pooling import last_token_pool, mean_pool
from cross_lingual_representations.retrieval import write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--block-layer", required=True, type=int)
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


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    dataset = load_flores_split(
        split=config.dataset.evaluation_split,
        languages=config.dataset.languages,
        n_samples=config.dataset.n_eval,
        seed=config.seed,
        dataset_name=config.dataset.name,
        dataset_config=config.dataset.config,
        text_field_template=config.dataset.text_field_template,
        id_field=config.dataset.id_field,
    )
    loaded = load_causal_lm(config.model)
    layers = _decoder_layers(loaded.model)
    block_index = args.block_layer - 1
    if block_index < 0 or block_index >= len(layers):
        raise ValueError(f"block-layer must be between 1 and {len(layers)}")
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
    stages = ("input", "post_attention", "output")
    pooling_methods = {"last_token": last_token_pool, "mean": mean_pool}
    by_key: dict[tuple[int, str, str], list[np.ndarray[Any, Any]]] = {
        (args.block_layer, stage, pooling): []
        for stage in stages
        for pooling in pooling_methods
    }
    try:
        for language, texts in dataset.sentences.items():
            language_values: dict[tuple[int, str, str], list[np.ndarray[Any, Any]]] = {
                key: [] for key in by_key
            }
            for text_batch in tqdm(
                _batches(texts, config.batch_size),
                total=(len(texts) + config.batch_size - 1) // config.batch_size,
                desc=f"block-{args.block_layer}/{language}",
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
                    for pooling, pooler in pooling_methods.items():
                        pooled = pooler(values, encoded["attention_mask"])
                        language_values[(args.block_layer, stage, pooling)].append(
                            pooled.to(device="cpu", dtype=torch.float32).numpy()
                        )
            for key in by_key:
                by_key[key].append(np.concatenate(language_values[key]))
    finally:
        for handle in handles:
            handle.remove()

    vectors = {
        key: np.stack(language_values).astype(np.float32)
        for key, language_values in by_key.items()
    }
    rows = block_stage_retrieval_rows(
        vectors,
        model_name=config.model.model_name,
        languages=tuple(dataset.sentences),
        sentence_ids=dataset.ids,
    )
    raw_path = (
        args.results_root
        / "raw"
        / f"{config.experiment_name}_block_{args.block_layer:02d}_retrieval.csv"
    )
    figure_path = (
        args.results_root
        / "figures"
        / f"{config.experiment_name}_block_{args.block_layer:02d}_decomposition.png"
    )
    write_tidy_csv(rows, raw_path)
    plot_block_stage_retrieval(rows, figure_path)
    print(f"Saved block decomposition: {raw_path}")


if __name__ == "__main__":
    main()

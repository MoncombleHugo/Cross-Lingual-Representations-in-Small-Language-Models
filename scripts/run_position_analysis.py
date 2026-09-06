"""Measure translation retrieval at relative token positions around selected layers."""

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
from cross_lingual_representations.diagnostics import (
    position_retrieval_rows,
    relative_position_pool,
)
from cross_lingual_representations.models import load_causal_lm
from cross_lingual_representations.plotting import plot_position_retrieval
from cross_lingual_representations.retrieval import write_tidy_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9])
    return parser.parse_args()


def _batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    layers = tuple(dict.fromkeys(args.layers))
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
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
    by_fraction: dict[float, list[np.ndarray[Any, Any]]] = {
        fraction: [] for fraction in fractions
    }
    for language, texts in dataset.sentences.items():
        language_values: dict[float, list[torch.Tensor]] = {
            fraction: [] for fraction in fractions
        }
        for text_batch in tqdm(
            _batches(texts, config.batch_size),
            total=(len(texts) + config.batch_size - 1) // config.batch_size,
            desc=f"positions/{language}",
        ):
            encoded = loaded.tokenizer(
                list(text_batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )
            encoded = {key: value.to(loaded.device) for key, value in encoded.items()}
            mask = encoded["attention_mask"]
            with torch.inference_mode():
                output = loaded.model(
                    **encoded, output_hidden_states=True, use_cache=False, return_dict=True
                )
            selected = relative_position_pool(
                tuple(output.hidden_states), mask, layers=layers, fractions=fractions
            )
            for fraction, values in selected.items():
                language_values[fraction].append(values.to(device="cpu", dtype=torch.float32))
        for fraction in fractions:
            by_fraction[fraction].append(torch.cat(language_values[fraction]).numpy())

    vectors = {
        fraction: np.stack(language_values).astype(np.float32)
        for fraction, language_values in by_fraction.items()
    }
    rows = position_retrieval_rows(
        vectors,
        model_name=config.model.model_name,
        languages=tuple(dataset.sentences),
        sentence_ids=dataset.ids,
        layers=layers,
    )
    raw_path = args.results_root / "raw" / f"{config.experiment_name}_position_retrieval.csv"
    figure_path = (
        args.results_root / "figures" / f"{config.experiment_name}_position_retrieval.png"
    )
    write_tidy_csv(rows, raw_path)
    plot_position_retrieval(rows, figure_path)
    print(f"Saved position-wise retrieval: {raw_path}")


if __name__ == "__main__":
    main()

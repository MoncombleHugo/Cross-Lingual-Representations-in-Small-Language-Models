"""Memory-efficient extraction of pooled sentence representations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
from tqdm.auto import tqdm

from cross_lingual_representations.cache import RepresentationBundle, utc_timestamp
from cross_lingual_representations.data import ParallelSplit
from cross_lingual_representations.models import LoadedModel, make_layer_labels
from cross_lingual_representations.pooling import PoolingMethod, pool_hidden_states

RepresentationDType = Literal["float16", "float32"]


def _batches(values: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def extract_representations(
    loaded: LoadedModel,
    dataset: ParallelSplit,
    *,
    pooling_methods: Sequence[PoolingMethod],
    batch_size: int,
    max_length: int,
    seed: int = 42,
    representation_dtype: RepresentationDType = "float16",
    show_progress: bool = True,
) -> dict[PoolingMethod, RepresentationBundle]:
    """Extract and immediately pool all layers, never retaining token states."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    methods = tuple(dict.fromkeys(pooling_methods))
    if not methods:
        raise ValueError("At least one pooling method is required")
    numpy_dtype = {"float16": np.float16, "float32": np.float32}[representation_dtype]
    by_method: dict[PoolingMethod, list[np.ndarray[Any, Any]]] = {
        method: [] for method in methods
    }
    observed_state_count: int | None = None

    for language, texts in dataset.sentences.items():
        language_batches: dict[PoolingMethod, list[torch.Tensor]] = {
            method: [] for method in methods
        }
        iterator = tqdm(
            _batches(texts, batch_size),
            total=(len(texts) + batch_size - 1) // batch_size,
            desc=f"extract {dataset.split}/{language}",
            disable=not show_progress,
        )
        for text_batch in iterator:
            encoded = loaded.tokenizer(
                list(text_batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {name: tensor.to(loaded.device) for name, tensor in encoded.items()}
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                raise ValueError("Tokenizer output is missing attention_mask")
            with torch.inference_mode():
                outputs = loaded.model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            hidden_states = outputs.hidden_states
            if hidden_states is None or len(hidden_states) == 0:
                raise RuntimeError("Model did not return hidden states")
            if observed_state_count is None:
                observed_state_count = len(hidden_states)
            elif len(hidden_states) != observed_state_count:
                raise RuntimeError("The number of hidden states changed between batches")
            for method in methods:
                pooled = pool_hidden_states(hidden_states, attention_mask, method)
                language_batches[method].append(pooled.to(device="cpu", dtype=torch.float32))
            del outputs, hidden_states

        for method in methods:
            language_tensor = torch.cat(language_batches[method], dim=0)
            by_method[method].append(language_tensor.numpy().astype(numpy_dtype, copy=False))

    if observed_state_count is None:
        raise RuntimeError("No representations were extracted")
    languages = tuple(dataset.sentences)
    language_codes = tuple(dataset.language_codes[language] for language in languages)
    created_at = utc_timestamp()
    return {
        method: RepresentationBundle(
            vectors=cast(Any, np.stack(language_vectors, axis=0)),
            model_name=loaded.config.model_name,
            model_revision=loaded.config.revision,
            split=dataset.split,
            languages=languages,
            language_codes=language_codes,
            sentence_ids=dataset.ids,
            layer_labels=make_layer_labels(observed_state_count),
            pooling=method,
            seed=seed,
            max_length=max_length,
            inference_dtype=str(loaded.dtype).removeprefix("torch."),
            representation_dtype=representation_dtype,
            created_at=created_at,
        )
        for method, language_vectors in by_method.items()
    }

"""Shared dataset/cache resolution for independently executable stages."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from cross_lingual_representations.cache import (
    CacheSpec,
    RepresentationBundle,
    load_representation_cache,
    representation_cache_path,
    validate_cache,
)
from cross_lingual_representations.config import ExperimentConfig
from cross_lingual_representations.data import ParallelSplit, load_flores_split

SplitRole = Literal["train", "evaluation"]


def load_configured_split(config: ExperimentConfig, role: SplitRole) -> ParallelSplit:
    """Load the deterministic train or evaluation sample declared in YAML."""
    split = config.dataset.train_split if role == "train" else config.dataset.evaluation_split
    n_samples = config.dataset.n_train if role == "train" else config.dataset.n_eval
    return load_flores_split(
        split=split,
        languages=config.dataset.languages,
        n_samples=n_samples,
        seed=config.seed,
        dataset_name=config.dataset.name,
        dataset_config=config.dataset.config,
        text_field_template=config.dataset.text_field_template,
        id_field=config.dataset.id_field,
    )


def load_configured_bundle(
    config: ExperimentConfig,
    cache_root: str | Path,
    *,
    role: SplitRole,
    pooling: str,
) -> RepresentationBundle:
    """Resolve and strictly validate a cache against its configured data sample."""
    dataset = load_configured_split(config, role)
    expected = CacheSpec(
        model_name=config.model.model_name,
        model_revision=config.model.revision,
        split=dataset.split,
        languages=tuple(dataset.sentences),
        language_codes=tuple(dataset.language_codes.values()),
        sentence_ids=dataset.ids,
        pooling=pooling,
        max_length=config.max_length,
        seed=config.seed,
        representation_dtype=config.representation_dtype,
    )
    path = representation_cache_path(cache_root, config.model.model_name, dataset.split, pooling)
    return load_representation_cache(path, expected=expected)


def load_configured_cached_bundle(
    config: ExperimentConfig,
    cache_root: str | Path,
    *,
    role: SplitRole,
    pooling: str,
) -> RepresentationBundle:
    """Validate a canonical cache without reopening its source dataset."""
    split = config.dataset.train_split if role == "train" else config.dataset.evaluation_split
    n_samples = config.dataset.n_train if role == "train" else config.dataset.n_eval
    path = representation_cache_path(cache_root, config.model.model_name, split, pooling)
    bundle = load_representation_cache(path)
    if len(bundle.sentence_ids) != n_samples:
        raise ValueError(
            f"Cached {split} sample count is {len(bundle.sentence_ids)}, expected {n_samples}"
        )
    validate_cache(
        bundle,
        CacheSpec(
            model_name=config.model.model_name,
            model_revision=config.model.revision,
            split=split,
            languages=tuple(config.dataset.languages),
            language_codes=tuple(config.dataset.languages.values()),
            sentence_ids=bundle.sentence_ids,
            pooling=pooling,
            max_length=config.max_length,
            seed=config.seed,
            representation_dtype=config.representation_dtype,
        ),
    )
    return bundle

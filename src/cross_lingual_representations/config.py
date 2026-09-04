"""Typed loading for the phase-oriented YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from cross_lingual_representations.models import DeviceChoice, DTypeChoice, ModelLoadConfig
from cross_lingual_representations.pooling import SUPPORTED_POOLING, PoolingMethod


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    config: str
    train_split: str
    evaluation_split: str
    n_train: int
    n_eval: int
    languages: dict[str, str]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_name: str
    seed: int
    model: ModelLoadConfig
    dataset: DatasetConfig
    pooling: tuple[PoolingMethod, ...]
    batch_size: int
    max_length: int
    representation_dtype: str = "float16"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate the subset of YAML used by phases 4-6."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "configuration")
    model_raw = _mapping(raw.get("model"), "model")
    dataset_raw = _mapping(raw.get("dataset"), "dataset")
    languages_raw = _mapping(dataset_raw.get("languages"), "dataset.languages")
    languages = {str(alias): str(code) for alias, code in languages_raw.items()}
    raw_pooling = raw.get("pooling")
    if not isinstance(raw_pooling, list) or not raw_pooling:
        raise ValueError("pooling must be a non-empty list")
    if any(method not in SUPPORTED_POOLING for method in raw_pooling):
        raise ValueError(f"pooling methods must be selected from {sorted(SUPPORTED_POOLING)}")
    pooling = cast(tuple[PoolingMethod, ...], tuple(dict.fromkeys(raw_pooling)))

    config = ExperimentConfig(
        experiment_name=str(raw["experiment_name"]),
        seed=int(raw["seed"]),
        model=ModelLoadConfig(
            model_name=str(model_raw["name"]),
            revision=model_raw.get("revision"),
            device=cast(DeviceChoice, model_raw.get("device", "auto")),
            dtype=cast(DTypeChoice, model_raw.get("dtype", "auto")),
            trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
        ),
        dataset=DatasetConfig(
            name=str(dataset_raw["name"]),
            config=str(dataset_raw.get("config", "all")),
            train_split=str(dataset_raw["train_split"]),
            evaluation_split=str(dataset_raw["evaluation_split"]),
            n_train=int(dataset_raw["n_train"]),
            n_eval=int(dataset_raw["n_eval"]),
            languages=languages,
        ),
        pooling=pooling,
        batch_size=int(raw["batch_size"]),
        max_length=int(raw["max_length"]),
        representation_dtype=str(raw.get("representation_dtype", "float16")),
    )
    if config.batch_size <= 0 or config.max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    if config.representation_dtype not in {"float16", "float32"}:
        raise ValueError("representation_dtype must be float16 or float32")
    return config

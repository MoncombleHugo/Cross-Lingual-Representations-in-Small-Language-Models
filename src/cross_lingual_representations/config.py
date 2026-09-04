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
    config: str | None
    train_split: str
    evaluation_split: str
    n_train: int
    n_eval: int
    languages: dict[str, str]
    text_field_template: str = "sentence_{code}"
    id_field: str | None = "id"


@dataclass(frozen=True, slots=True)
class ProcrustesConfig:
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class LanguageProbeConfig:
    enabled: bool = True
    c: float = 1.0
    max_iter: int = 2000


@dataclass(frozen=True, slots=True)
class TokenizationConfig:
    enabled: bool = True
    add_special_tokens: bool = True


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
    procrustes: ProcrustesConfig = ProcrustesConfig()
    language_probe: LanguageProbeConfig = LanguageProbeConfig()
    tokenization: TokenizationConfig = TokenizationConfig()


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
    procrustes_raw = _mapping(raw.get("procrustes", {}), "procrustes")
    probe_raw = _mapping(raw.get("language_probe", {}), "language_probe")
    tokenization_raw = _mapping(raw.get("tokenization", {}), "tokenization")
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
            config=(
                None
                if dataset_raw.get("config", "all") is None
                else str(dataset_raw.get("config", "all"))
            ),
            train_split=str(dataset_raw["train_split"]),
            evaluation_split=str(dataset_raw["evaluation_split"]),
            n_train=int(dataset_raw["n_train"]),
            n_eval=int(dataset_raw["n_eval"]),
            languages=languages,
            text_field_template=str(
                dataset_raw.get("text_field_template", "sentence_{code}")
            ),
            id_field=(
                None
                if dataset_raw.get("id_field", "id") is None
                else str(dataset_raw.get("id_field", "id"))
            ),
        ),
        pooling=pooling,
        batch_size=int(raw["batch_size"]),
        max_length=int(raw["max_length"]),
        representation_dtype=str(raw.get("representation_dtype", "float16")),
        procrustes=ProcrustesConfig(enabled=bool(procrustes_raw.get("enabled", True))),
        language_probe=LanguageProbeConfig(
            enabled=bool(probe_raw.get("enabled", True)),
            c=float(probe_raw.get("c", 1.0)),
            max_iter=int(probe_raw.get("max_iter", 2000)),
        ),
        tokenization=TokenizationConfig(
            enabled=bool(tokenization_raw.get("enabled", True)),
            add_special_tokens=bool(tokenization_raw.get("add_special_tokens", True)),
        ),
    )
    if config.batch_size <= 0 or config.max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    if config.representation_dtype not in {"float16", "float32"}:
        raise ValueError("representation_dtype must be float16 or float32")
    if config.language_probe.c <= 0 or config.language_probe.max_iter <= 0:
        raise ValueError("language_probe c and max_iter must be positive")
    return config

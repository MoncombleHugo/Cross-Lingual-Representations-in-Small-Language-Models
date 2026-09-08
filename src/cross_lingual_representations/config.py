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
class NonParallelCenteringConfig:
    """Unlabelled monolingual corpus used to estimate language centroids."""

    enabled: bool = False
    dataset_name: str = "allenai/c4"
    dataset_configs: dict[str, str] | None = None
    split: str = "train"
    text_field: str = "text"
    id_field: str | None = "id"
    revision: str | None = None
    pool_size: int = 1024
    shuffle_buffer_size: int = 10_000
    sample_sizes: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
    seeds: tuple[int, ...] = tuple(range(10))


@dataclass(frozen=True, slots=True)
class CrosslingualTransferConfig:
    """Frozen-representation multilingual intent-transfer experiment."""

    enabled: bool = False
    dataset_name: str = "AmazonScience/massive"
    data_file_template: str | None = (
        "https://huggingface.co/datasets/{dataset}/resolve/"
        "refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet"
    )
    dataset_configs: dict[str, str] | None = None
    train_split: str = "train"
    evaluation_split: str = "test"
    text_field: str = "utt"
    label_field: str = "intent"
    id_field: str = "id"
    n_train: int = 3000
    n_eval: int = 1500
    layers: tuple[str | int, ...] = (
        "embedding",
        "best_raw",
        "intermediate",
        "final",
    )
    batch_size: int = 64
    max_length: int = 64
    representation_dtype: str = "float32"
    c: float = 1.0
    max_iter: int = 2000


@dataclass(frozen=True, slots=True)
class LanguageSubspaceConfig:
    """Centroid-derived language-subspace removal experiment."""

    enabled: bool = False
    dimensions: tuple[int, ...] = (1, 2, 3)
    random_seeds: tuple[int, ...] = (42,)
    c: float = 1.0
    max_iter: int = 2000


@dataclass(frozen=True, slots=True)
class BlockMechanismConfig:
    """Residual-stream mechanism analysis at one selected decoder block."""

    enabled: bool = False
    block_layer: int = 1
    subspace_dimensions: int = 3
    c: float = 1.0
    max_iter: int = 2000


@dataclass(frozen=True, slots=True)
class ResidualInterventionConfig:
    """Causal removal of train-derived directions from a block output."""

    enabled: bool = False
    basis_pooling: str = "mean"
    dimensions: int = 3
    random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    c: float = 1.0
    max_iter: int = 2000


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
    nonparallel_centering: NonParallelCenteringConfig = NonParallelCenteringConfig()
    crosslingual_transfer: CrosslingualTransferConfig = CrosslingualTransferConfig()
    language_subspace: LanguageSubspaceConfig = LanguageSubspaceConfig()
    block_mechanism: BlockMechanismConfig = BlockMechanismConfig()
    residual_intervention: ResidualInterventionConfig = ResidualInterventionConfig()


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
    centering_raw = _mapping(raw.get("nonparallel_centering", {}), "nonparallel_centering")
    transfer_raw = _mapping(raw.get("crosslingual_transfer", {}), "crosslingual_transfer")
    subspace_raw = _mapping(raw.get("language_subspace", {}), "language_subspace")
    mechanism_raw = _mapping(raw.get("block_mechanism", {}), "block_mechanism")
    intervention_raw = _mapping(raw.get("residual_intervention", {}), "residual_intervention")
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
            text_field_template=str(dataset_raw.get("text_field_template", "sentence_{code}")),
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
        nonparallel_centering=NonParallelCenteringConfig(
            enabled=bool(centering_raw.get("enabled", False)),
            dataset_name=str(centering_raw.get("dataset_name", "allenai/c4")),
            dataset_configs={
                str(language): str(dataset_config)
                for language, dataset_config in _mapping(
                    centering_raw.get("dataset_configs", languages),
                    "nonparallel_centering.dataset_configs",
                ).items()
            },
            split=str(centering_raw.get("split", "train")),
            text_field=str(centering_raw.get("text_field", "text")),
            id_field=(
                None
                if centering_raw.get("id_field", "id") is None
                else str(centering_raw.get("id_field", "id"))
            ),
            revision=(
                None if centering_raw.get("revision") is None else str(centering_raw["revision"])
            ),
            pool_size=int(centering_raw.get("pool_size", 1024)),
            shuffle_buffer_size=int(centering_raw.get("shuffle_buffer_size", 10_000)),
            sample_sizes=tuple(
                int(value)
                for value in centering_raw.get("sample_sizes", (8, 16, 32, 64, 128, 256, 512))
            ),
            seeds=tuple(int(value) for value in centering_raw.get("seeds", range(10))),
        ),
        crosslingual_transfer=CrosslingualTransferConfig(
            enabled=bool(transfer_raw.get("enabled", False)),
            dataset_name=str(transfer_raw.get("dataset_name", "AmazonScience/massive")),
            data_file_template=(
                None
                if transfer_raw.get("data_file_template", "default") is None
                else str(
                    transfer_raw.get(
                        "data_file_template",
                        "https://huggingface.co/datasets/{dataset}/resolve/"
                        "refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet",
                    )
                )
            ),
            dataset_configs={
                str(language): str(dataset_config)
                for language, dataset_config in _mapping(
                    transfer_raw.get(
                        "dataset_configs",
                        {"en": "en-US", "ko": "ko-KR", "ja": "ja-JP", "zh": "zh-CN"},
                    ),
                    "crosslingual_transfer.dataset_configs",
                ).items()
            },
            train_split=str(transfer_raw.get("train_split", "train")),
            evaluation_split=str(transfer_raw.get("evaluation_split", "test")),
            text_field=str(transfer_raw.get("text_field", "utt")),
            label_field=str(transfer_raw.get("label_field", "intent")),
            id_field=str(transfer_raw.get("id_field", "id")),
            n_train=int(transfer_raw.get("n_train", 3000)),
            n_eval=int(transfer_raw.get("n_eval", 1500)),
            layers=tuple(
                transfer_raw.get("layers", ("embedding", "best_raw", "intermediate", "final"))
            ),
            batch_size=int(transfer_raw.get("batch_size", 64)),
            max_length=int(transfer_raw.get("max_length", 64)),
            representation_dtype=str(transfer_raw.get("representation_dtype", "float32")),
            c=float(transfer_raw.get("c", 1.0)),
            max_iter=int(transfer_raw.get("max_iter", 2000)),
        ),
        language_subspace=LanguageSubspaceConfig(
            enabled=bool(subspace_raw.get("enabled", False)),
            dimensions=tuple(int(value) for value in subspace_raw.get("dimensions", (1, 2, 3))),
            random_seeds=tuple(int(value) for value in subspace_raw.get("random_seeds", (42,))),
            c=float(subspace_raw.get("c", 1.0)),
            max_iter=int(subspace_raw.get("max_iter", 2000)),
        ),
        block_mechanism=BlockMechanismConfig(
            enabled=bool(mechanism_raw.get("enabled", False)),
            block_layer=int(mechanism_raw.get("block_layer", 1)),
            subspace_dimensions=int(mechanism_raw.get("subspace_dimensions", 3)),
            c=float(mechanism_raw.get("c", 1.0)),
            max_iter=int(mechanism_raw.get("max_iter", 2000)),
        ),
        residual_intervention=ResidualInterventionConfig(
            enabled=bool(intervention_raw.get("enabled", False)),
            basis_pooling=str(intervention_raw.get("basis_pooling", "mean")),
            dimensions=int(intervention_raw.get("dimensions", 3)),
            random_seeds=tuple(
                int(value) for value in intervention_raw.get("random_seeds", (0, 1, 2, 3, 4))
            ),
            c=float(intervention_raw.get("c", 1.0)),
            max_iter=int(intervention_raw.get("max_iter", 2000)),
        ),
    )
    if config.batch_size <= 0 or config.max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    if config.representation_dtype not in {"float16", "float32"}:
        raise ValueError("representation_dtype must be float16 or float32")
    if config.language_probe.c <= 0 or config.language_probe.max_iter <= 0:
        raise ValueError("language_probe c and max_iter must be positive")
    centering = config.nonparallel_centering
    if centering.enabled:
        if config.dataset.name == centering.dataset_name:
            raise ValueError("nonparallel centroids must use a dataset distinct from evaluation")
        if centering.dataset_configs is None or set(centering.dataset_configs) != set(languages):
            raise ValueError("nonparallel dataset_configs must define every configured language")
        if centering.pool_size <= 0 or centering.shuffle_buffer_size <= 0:
            raise ValueError("nonparallel pool_size and shuffle_buffer_size must be positive")
        if not centering.sample_sizes or any(value <= 0 for value in centering.sample_sizes):
            raise ValueError("nonparallel sample_sizes must contain positive values")
        if max(centering.sample_sizes) > centering.pool_size:
            raise ValueError("nonparallel sample_sizes cannot exceed pool_size")
        if not centering.seeds or len(set(centering.seeds)) != len(centering.seeds):
            raise ValueError("nonparallel seeds must be non-empty and unique")
    transfer = config.crosslingual_transfer
    if transfer.enabled:
        if transfer.dataset_configs is None or set(transfer.dataset_configs) != set(languages):
            raise ValueError("crosslingual transfer dataset_configs must define every language")
        if transfer.train_split == transfer.evaluation_split:
            raise ValueError("crosslingual transfer train and evaluation splits must differ")
        if min(transfer.n_train, transfer.n_eval, transfer.batch_size, transfer.max_length) <= 0:
            raise ValueError("crosslingual transfer sizes must be positive")
        if transfer.c <= 0 or transfer.max_iter <= 0:
            raise ValueError("crosslingual transfer classifier parameters must be positive")
        if transfer.representation_dtype not in {"float16", "float32"}:
            raise ValueError(
                "crosslingual transfer representation_dtype must be float16 or float32"
            )
        valid_layer_names = {"embedding", "best_raw", "intermediate", "final"}
        if not transfer.layers or any(
            not isinstance(layer, int) and layer not in valid_layer_names
            for layer in transfer.layers
        ):
            raise ValueError("crosslingual transfer layers contain an invalid selector")
    subspace = config.language_subspace
    if subspace.enabled:
        max_dimension = len(languages) - 1
        if not subspace.dimensions or any(
            value <= 0 or value > max_dimension for value in subspace.dimensions
        ):
            raise ValueError("language_subspace dimensions must be between 1 and languages - 1")
        if len(set(subspace.dimensions)) != len(subspace.dimensions):
            raise ValueError("language_subspace dimensions must be unique")
        if not subspace.random_seeds or len(set(subspace.random_seeds)) != len(
            subspace.random_seeds
        ):
            raise ValueError("language_subspace random_seeds must be non-empty and unique")
        if subspace.c <= 0 or subspace.max_iter <= 0:
            raise ValueError("language_subspace classifier parameters must be positive")
    mechanism = config.block_mechanism
    if mechanism.enabled:
        if mechanism.block_layer <= 0:
            raise ValueError("block_mechanism block_layer must be positive")
        if not 0 < mechanism.subspace_dimensions <= len(languages) - 1:
            raise ValueError(
                "block_mechanism subspace_dimensions must be between 1 and languages - 1"
            )
        if mechanism.c <= 0 or mechanism.max_iter <= 0:
            raise ValueError("block_mechanism classifier parameters must be positive")
    intervention = config.residual_intervention
    if intervention.enabled:
        if not config.block_mechanism.enabled:
            raise ValueError("residual_intervention requires block_mechanism")
        if not 0 < intervention.dimensions <= len(languages) - 1:
            raise ValueError("residual_intervention dimensions must be between 1 and languages - 1")
        if intervention.basis_pooling not in config.pooling:
            raise ValueError("residual_intervention basis_pooling must be configured in pooling")
        if not intervention.random_seeds or len(set(intervention.random_seeds)) != len(
            intervention.random_seeds
        ):
            raise ValueError("residual_intervention random_seeds must be non-empty and unique")
        if intervention.c <= 0 or intervention.max_iter <= 0:
            raise ValueError("residual_intervention classifier parameters must be positive")
    return config

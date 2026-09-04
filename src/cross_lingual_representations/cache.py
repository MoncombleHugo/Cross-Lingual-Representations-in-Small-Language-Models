"""Validated, non-pickle representation caches."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CacheMismatchError(ValueError):
    """Raised when a cache does not describe the requested experiment."""


@dataclass(frozen=True, slots=True)
class CacheSpec:
    """Fields that must match before a representation cache may be reused."""

    model_name: str
    model_revision: str | None
    split: str
    languages: tuple[str, ...]
    language_codes: tuple[str, ...]
    sentence_ids: tuple[str, ...]
    pooling: str
    max_length: int
    seed: int
    representation_dtype: str


@dataclass(frozen=True, slots=True)
class RepresentationBundle:
    """Pooled vectors and the metadata needed to interpret them safely."""

    vectors: NDArray[np.floating[Any]]
    model_name: str
    model_revision: str | None
    split: str
    languages: tuple[str, ...]
    language_codes: tuple[str, ...]
    sentence_ids: tuple[str, ...]
    layer_labels: tuple[str, ...]
    pooling: str
    seed: int
    max_length: int
    inference_dtype: str
    representation_dtype: str
    created_at: str

    def __post_init__(self) -> None:
        if self.vectors.ndim != 4:
            raise ValueError("vectors must have shape [languages, sentences, states, hidden]")
        expected = (
            len(self.languages),
            len(self.sentence_ids),
            len(self.layer_labels),
        )
        if self.vectors.shape[:3] != expected:
            raise ValueError(
                f"vectors start with shape {self.vectors.shape[:3]}, expected {expected}"
            )
        if len(self.language_codes) != len(self.languages):
            raise ValueError("language_codes and languages must have the same length")
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("languages must be unique")
        if len(set(self.sentence_ids)) != len(self.sentence_ids):
            raise ValueError("sentence_ids must be unique")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    @property
    def spec(self) -> CacheSpec:
        """Return the compatibility-critical cache fields."""
        return CacheSpec(
            model_name=self.model_name,
            model_revision=self.model_revision,
            split=self.split,
            languages=self.languages,
            language_codes=self.language_codes,
            sentence_ids=self.sentence_ids,
            pooling=self.pooling,
            max_length=self.max_length,
            seed=self.seed,
            representation_dtype=self.representation_dtype,
        )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible metadata, excluding the vector payload."""
        metadata = asdict(self)
        del metadata["vectors"]
        return metadata


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for cache provenance."""
    return datetime.now(UTC).isoformat()


def model_cache_name(model_name: str) -> str:
    """Convert a Hub model identifier to a filesystem-safe directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_name).strip("._-")


def representation_cache_path(
    root: str | Path,
    model_name: str,
    split: str,
    pooling: str,
) -> Path:
    """Build the canonical representation-cache path."""
    return Path(root) / model_cache_name(model_name) / split / f"{pooling}.npz"


def save_representation_cache(bundle: RepresentationBundle, path: str | Path) -> Path:
    """Atomically save vectors and JSON metadata without Python pickles."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    metadata_json = json.dumps(bundle.metadata(), ensure_ascii=False, sort_keys=True)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, vectors=bundle.vectors, metadata=np.array(metadata_json))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _metadata_to_bundle(
    vectors: NDArray[np.floating[Any]], metadata: dict[str, Any]
) -> RepresentationBundle:
    revision = metadata["model_revision"]
    return RepresentationBundle(
        vectors=vectors,
        model_name=str(metadata["model_name"]),
        model_revision=None if revision is None else str(revision),
        split=str(metadata["split"]),
        languages=tuple(str(value) for value in metadata["languages"]),
        language_codes=tuple(str(value) for value in metadata["language_codes"]),
        sentence_ids=tuple(str(value) for value in metadata["sentence_ids"]),
        layer_labels=tuple(str(value) for value in metadata["layer_labels"]),
        pooling=str(metadata["pooling"]),
        seed=int(metadata["seed"]),
        max_length=int(metadata["max_length"]),
        inference_dtype=str(metadata["inference_dtype"]),
        representation_dtype=str(metadata["representation_dtype"]),
        created_at=str(metadata["created_at"]),
    )


def validate_cache(bundle: RepresentationBundle, expected: CacheSpec) -> None:
    """Reject every compatibility-critical mismatch with a readable error."""
    mismatches: list[str] = []
    for field, expected_value in asdict(expected).items():
        actual_value = getattr(bundle, field)
        if actual_value != expected_value:
            mismatches.append(f"{field}: expected {expected_value!r}, found {actual_value!r}")
    if mismatches:
        raise CacheMismatchError("Incompatible representation cache:\n- " + "\n- ".join(mismatches))


def load_representation_cache(
    path: str | Path,
    *,
    expected: CacheSpec | None = None,
) -> RepresentationBundle:
    """Load a cache with ``allow_pickle=False`` and optionally validate it."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"vectors", "metadata"}:
            raise ValueError(f"Unexpected arrays in cache {source}: {archive.files}")
        vectors = archive["vectors"]
        raw_metadata = archive["metadata"]
        if raw_metadata.ndim != 0:
            raise ValueError("Cache metadata must be a scalar JSON string")
        metadata = json.loads(str(raw_metadata.item()))
    if not isinstance(metadata, dict):
        raise ValueError("Cache metadata must decode to an object")
    bundle = _metadata_to_bundle(vectors, metadata)
    if expected is not None:
        validate_cache(bundle, expected)
    return bundle

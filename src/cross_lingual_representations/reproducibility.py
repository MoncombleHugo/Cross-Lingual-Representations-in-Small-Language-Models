"""Reproducibility helpers shared by experiment scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and all visible CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_selected_ids(
    path: str | Path,
    *,
    split: str,
    seed: int,
    sentence_ids: tuple[str, ...],
) -> Path:
    """Persist the exact sampled IDs separately from vector caches."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "seed": seed, "sentence_ids": list(sentence_ids)}
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


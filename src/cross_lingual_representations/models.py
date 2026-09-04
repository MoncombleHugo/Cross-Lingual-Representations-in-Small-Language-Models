"""Generic Hugging Face causal language-model loading and inspection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast

import torch

DeviceChoice = Literal["auto", "cpu", "cuda"]
DTypeChoice = Literal["auto", "float32", "float16", "bfloat16"]


class TokenizerLike(Protocol):
    """Tokenizer surface used by model loading and inspection."""

    padding_side: str
    pad_token: str | None
    eos_token: str | None
    pad_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None

    def __call__(
        self, text: str | Sequence[str], **kwargs: Any
    ) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class ModelLoadConfig:
    """Configuration for a reproducible causal-LM load."""

    model_name: str
    revision: str | None = None
    device: DeviceChoice = "auto"
    dtype: DTypeChoice = "auto"
    trust_remote_code: bool = False


@dataclass(slots=True)
class LoadedModel:
    """A model/tokenizer pair plus resolved runtime metadata."""

    model: Any
    tokenizer: TokenizerLike
    device: torch.device
    dtype: torch.dtype
    parameter_count: int
    config: ModelLoadConfig


@dataclass(frozen=True, slots=True)
class HiddenStateInspection:
    """Observed hidden-state and tokenizer metadata for one sentence."""

    model_name: str
    device: str
    dtype: str
    parameter_count: int
    input_tokens: int
    num_hidden_states: int
    hidden_size: int
    layer_labels: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    special_tokens: dict[str, int | None]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""
        result = asdict(self)
        result["layer_labels"] = list(self.layer_labels)
        result["shapes"] = [list(shape) for shape in self.shapes]
        return result


def resolve_device(choice: DeviceChoice) -> torch.device:
    """Resolve ``auto`` without silently accepting unavailable CUDA."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(choice)


def resolve_dtype(choice: DTypeChoice, device: torch.device) -> torch.dtype:
    """Choose a conservative inference dtype for the resolved device."""
    if choice != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[choice]
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _pretrained_kwargs(config: ModelLoadConfig) -> dict[str, Any]:
    common_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
    }
    return {key: value for key, value in common_kwargs.items() if value is not None}


def load_tokenizer(config: ModelLoadConfig) -> TokenizerLike:
    """Load and configure the tokenizer without loading model weights."""
    from transformers import AutoTokenizer

    tokenizer = cast(
        TokenizerLike,
        AutoTokenizer.from_pretrained(config.model_name, **_pretrained_kwargs(config)),
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a padding token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(config: ModelLoadConfig) -> LoadedModel:
    """Load a base causal LM sequentially with cache disabled for inspection."""
    from transformers import AutoModelForCausalLM

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    common_kwargs = _pretrained_kwargs(config)
    tokenizer = load_tokenizer(config)

    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=dtype,
            **common_kwargs,
        ),
    )
    model.config.use_cache = False
    model.eval()
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return LoadedModel(model, tokenizer, device, dtype, parameter_count, config)


def make_layer_labels(num_hidden_states: int) -> tuple[str, ...]:
    """Name the embedding output and however many block outputs were observed."""
    if num_hidden_states <= 0:
        raise ValueError("num_hidden_states must be positive")
    return (
        "layer_00_embedding",
        *(f"layer_{index:02d}" for index in range(1, num_hidden_states)),
    )


def inspect_hidden_states(
    loaded: LoadedModel,
    text: str,
    *,
    max_length: int = 128,
) -> HiddenStateInspection:
    """Run one sentence and validate the model's observed hidden-state convention."""
    if not text.strip():
        raise ValueError("Inspection text must not be empty")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    encoded = loaded.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {name: tensor.to(loaded.device) for name, tensor in encoded.items()}
    if "input_ids" not in encoded:
        raise ValueError("Tokenizer output is missing input_ids")

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

    shapes = tuple(tuple(int(size) for size in state.shape) for state in hidden_states)
    if any(len(shape) != 3 for shape in shapes):
        raise RuntimeError(f"Expected [batch, tokens, hidden] states, observed {shapes}")
    if len({shape[-1] for shape in shapes}) != 1:
        raise RuntimeError(f"Hidden size changes between states: {shapes}")

    tokenizer = loaded.tokenizer
    return HiddenStateInspection(
        model_name=loaded.config.model_name,
        device=str(loaded.device),
        dtype=str(loaded.dtype).removeprefix("torch."),
        parameter_count=loaded.parameter_count,
        input_tokens=int(encoded["input_ids"].shape[1]),
        num_hidden_states=len(hidden_states),
        hidden_size=shapes[0][-1],
        layer_labels=make_layer_labels(len(hidden_states)),
        shapes=shapes,
        special_tokens={
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
    )

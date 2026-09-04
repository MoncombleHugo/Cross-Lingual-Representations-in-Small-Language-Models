"""Load a causal LM, run one sentence, and print observed hidden-state metadata."""

from __future__ import annotations

import argparse
import json

from cross_lingual_representations.models import (
    ModelLoadConfig,
    inspect_hidden_states,
    load_causal_lm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="trillionlabs/Tri-0.5B-Base")
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--text", default="The train arrives at six.")
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelLoadConfig(
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    loaded = load_causal_lm(config)
    inspection = inspect_hidden_states(loaded, args.text, max_length=args.max_length)
    print(json.dumps(inspection.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


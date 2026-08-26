"""Pre-cache the frozen P0p models and RefineBench revision on a CPU node."""

from __future__ import annotations

import argparse
from pathlib import Path

from policy_representation_swap_p0p import (
    DATASET_NAME,
    DATASET_REVISION,
    DEFAULT_JUDGE,
    MODEL_SPECS,
)


MODEL_FILE_PATTERNS = [
    "*.json",
    "*.jinja",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.tiktoken",
    "*.txt",
]


def run(hf_home: Path, dataset_cache: Path) -> None:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    hf_home.mkdir(parents=True, exist_ok=True)
    dataset_cache.mkdir(parents=True, exist_ok=True)
    model_ids = [*MODEL_SPECS.values(), DEFAULT_JUDGE]
    for model_id in model_ids:
        print(f"[cache-model] {model_id}", flush=True)
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(hf_home),
            allow_patterns=MODEL_FILE_PATTERNS,
        )
    print(f"[cache-dataset] {DATASET_NAME}@{DATASET_REVISION}", flush=True)
    dataset = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=str(dataset_cache),
    )
    print(f"CACHE_DONE rows={len(dataset)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.hf_home, args.dataset_cache)

#!/usr/bin/env python3
"""
Reshard safetensors checkpoint files to smaller shards.

Usage:
    python tools/reshard_safetensors.py <input_dir> [--output_dir <output_dir>] [--max_shard_size 5GB]

Examples:
    # Reshard in-place (replaces original files):
    python tools/reshard_safetensors.py saves/my_checkpoint

    # Reshard to a new directory:
    python tools/reshard_safetensors.py saves/my_checkpoint --output_dir saves/my_checkpoint_sharded

    # Custom shard size:
    python tools/reshard_safetensors.py saves/my_checkpoint --max_shard_size 4GB
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def parse_size(size_str: str) -> int:
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    size_str = size_str.strip().upper()
    for unit, multiplier in sorted(units.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(unit):
            return int(float(size_str[: -len(unit)]) * multiplier)
    return int(size_str)


def get_tensor_size(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def reshard(input_dir: str, output_dir: str, max_shard_size: int):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    index_file = input_path / "model.safetensors.index.json"
    single_file = input_path / "model.safetensors"

    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        print(f"Found sharded model with {len(shard_files)} shards")
    elif single_file.exists():
        shard_files = ["model.safetensors"]
        print("Found single-file model")
    else:
        print(f"Error: No model.safetensors or model.safetensors.index.json in {input_dir}")
        sys.exit(1)

    print(f"Loading all tensors from {input_dir}...")
    all_tensors = {}
    for shard_name in shard_files:
        shard_path = input_path / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    total_size = sum(get_tensor_size(t) for t in all_tensors.values())
    print(f"Total model size: {total_size / 1024**3:.2f} GB, {len(all_tensors)} tensors")

    # Split into shards
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted(all_tensors.keys()):
        tensor = all_tensors[key]
        tensor_size = get_tensor_size(tensor)

        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    print(f"Will create {num_shards} shards (max_shard_size={max_shard_size / 1024**3:.1f} GB)")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save shards
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        shard_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shard_path = output_path / shard_name
        print(f"  Saving {shard_name} ({len(shard)} tensors, {sum(get_tensor_size(t) for t in shard.values()) / 1024**3:.2f} GB)")
        save_file(shard, str(shard_path), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = shard_name

    # Write index
    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_output = output_path / "model.safetensors.index.json"
    with open(index_output, "w") as f:
        json.dump(index_data, f, indent=2)

    # Copy non-safetensors files (config, tokenizer, etc.)
    if str(input_path.resolve()) != str(output_path.resolve()):
        for item in input_path.iterdir():
            if item.is_file() and not item.name.endswith(".safetensors") and item.name != "model.safetensors.index.json":
                dest = output_path / item.name
                if not dest.exists():
                    shutil.copy2(str(item), str(dest))
                    print(f"  Copied {item.name}")

    print(f"Done! {num_shards} shards saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Reshard safetensors checkpoint to smaller files")
    parser.add_argument("input_dir", help="Path to the checkpoint directory")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: same as input, in-place)")
    parser.add_argument("--max_shard_size", default="5GB", help="Max shard size (default: 5GB)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    max_shard_size = parse_size(args.max_shard_size)

    if output_dir == args.input_dir:
        print(f"WARNING: In-place reshard will replace original files in {args.input_dir}")
        print("Press Ctrl+C to cancel, or Enter to continue...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    reshard(args.input_dir, output_dir, max_shard_size)


if __name__ == "__main__":
    main()

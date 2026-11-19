# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[2]


def validate_topology(config):
    from transformers import TrainingArguments

    def positive_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    nodes = positive_int("NNODES", int(os.environ.get("NNODES", "1")))
    world = nodes * positive_int("NPROC_PER_NODE", int(os.environ.get("NPROC_PER_NODE", "8")))
    rank = int(os.environ.get("NODE_RANK", "0"))
    if not 0 <= rank < nodes:
        raise ValueError("NODE_RANK must be in [0, NNODES)")
    if nodes > 1 and os.environ.get("MASTER_ADDR", "") in {"", "localhost", "127.0.0.1"}:
        raise ValueError("Multi-node training requires a reachable MASTER_ADDR")
    # Match the adapter's precedence: explicit arguments > additional_configs > checkpoint > defaults.
    inherited = {}
    if config.get("model_name_or_path"):
        checkpoint_config = Path(config["model_name_or_path"]) / "mca_config.json"
        if checkpoint_config.is_file():
            inherited = json.loads(checkpoint_config.read_text())
    inherited.update(config.get("additional_configs") or {})
    tp, pp, cp = [
        positive_int(name, config[name] if config.get(name) is not None else inherited.get(name, 1))
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")
    ]
    if world % (tp * pp * cp):
        raise ValueError(f"WORLD_SIZE={world} must be divisible by TP*PP*CP={tp * pp * cp}")
    dp = world // (tp * pp * cp)
    # MCA inherits these defaults from HF; do not construct arguments or initialize devices.
    defaults = {field.name: field.default for field in dataclasses.fields(TrainingArguments)}
    micro = positive_int(
        "per_device_train_batch_size",
        config.get("per_device_train_batch_size", defaults["per_device_train_batch_size"]),
    )
    accum = positive_int(
        "gradient_accumulation_steps",
        config.get("gradient_accumulation_steps", defaults["gradient_accumulation_steps"]),
    )
    if config.get("pretrain_score_only") and not config.get("pretrain_score"):
        raise ValueError("pretrain_score_only requires pretrain_score")
    if config.get("pretrain_score") and (cp != 1 or dp != 1 or config.get("resume_from_checkpoint")):
        raise ValueError("pretrain_score requires fresh CP1/DP1 training")
    if (config.get("neat_packing") or config.get("sequence_packing")) and cp > 1:
        alignment = int(os.environ.get("LF_PACK_SUBSEQ_ALIGN", "0"))
        if alignment <= 0 or alignment % (2 * cp):
            raise ValueError("CP packing requires LF_PACK_SUBSEQ_ALIGN to be a positive multiple of 2*CP")
    print(f"[batch] WORLD_SIZE={world} TP={tp} PP={pp} CP={cp} DP={dp} GBS={micro * accum * dp}")


def validate_input_paths(config):
    model = Path(config["model_name_or_path"])
    if not (model / "config.json").is_file():
        raise ValueError("Set model_name_or_path to an existing local HF checkpoint")
    tokenized = config.get("tokenized_path")
    if tokenized:
        if not Path(tokenized).is_dir():
            raise ValueError(f"tokenized_path does not exist: {tokenized}")
    else:
        data_dir = Path(config["dataset_dir"])
        info = json.loads((data_dir / "dataset_info.json").read_text())
        for name in config["dataset"].split(","):
            entry = info[name.strip()]
            if "file_name" in entry and not (data_dir / entry["file_name"]).is_file():
                raise ValueError(f"Missing dataset file for {name}")
    # Node 0 may create the shared output while other nodes are still in preflight.
    if int(os.environ.get("NODE_RANK", "0")) == 0:
        if Path(config["output_dir"]).exists() and not config.get("resume_from_checkpoint"):
            raise ValueError("Choose a new output directory; preflight refuses existing run directories")


def validate_versions():
    expected = {
        "torch": "2.10.0+cu128",
        "transformers": "5.6.0",
        "megatron-core": "0.18.2",
        "transformer-engine": "2.18.0",
        "flash-attn": "2.8.3.post1",
        "flash-linear-attention": "0.5.1",
        "fla-core": "0.5.1",
        "nvidia-cudnn-cu12": "9.26.0.51",
    }
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        # The 9.10 THD correctness issue is in cuDNN attention, not FlashAttention.
        flash_only_cudnn = (
            name == "nvidia-cudnn-cu12"
            and actual == "9.10.2.21"
            and os.environ.get("NVTE_FLASH_ATTN") == "1"
            and os.environ.get("NVTE_FUSED_ATTN") == "0"
        )
        if actual != version and not flash_only_cudnn:
            raise ValueError(f"{name}: expected {version}, got {actual}")
        if flash_only_cudnn:
            print("[env] cuDNN 9.10.2.21 accepted with FlashAttention enabled and cuDNN attention disabled")


def main():
    # Transformers may probe CUDA during import; hide GPUs before any third-party imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description="CPU-only dependency/config checks; no model loading or training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument(
        "--config-only", action="store_true", help="Validate topology without importing the MCA training stack"
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from omegaconf import OmegaConf

    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    config = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides)), resolve=True
    )
    validate_topology(config)
    if args.config_only:
        print("Config-only validation passed; dependencies, data and GPU execution were not checked")
        return
    os.environ["USE_MCA"] = "1"
    validate_versions()
    for name, directory in {
        "mcore_adapter": ROOT / "src/mcore_adapter",
        "llamafactory": ROOT / "src/llamafactory",
    }.items():
        spec = importlib.util.find_spec(name)
        if spec is None or Path(spec.origin).resolve().parent != directory:
            raise ValueError(f"{name} resolves outside this checkout: {spec.origin if spec else None}")
    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    from mcore_adapter.models.qwen3_5.config_qwen3_5 import Qwen3_5Config
    from mcore_adapter.training_args import Seq2SeqTrainingArguments

    fields = set()
    for cls in [Seq2SeqTrainingArguments, DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments]:
        fields.update(field.name for field in dataclasses.fields(cls))
    unknown = config.keys() - fields
    if unknown:
        raise ValueError(f"Unknown training fields: {sorted(unknown)}")
    data_fields = {field.name for field in dataclasses.fields(DataArguments)}
    DataArguments(**{key: value for key, value in config.items() if key in data_fields})
    model_fields = {field.name for field in dataclasses.fields(Qwen3_5Config)}
    unknown = config.get("additional_configs", {}).keys() - model_fields
    if unknown:
        raise ValueError(f"Unknown model fields: {sorted(unknown)}")
    if args.check_data:
        validate_input_paths(config)
    print("CPU preflight passed: imports, versions, source identity, and all YAML/model fields")
    print("This does not replace a multi-GPU optimizer-step acceptance run")


if __name__ == "__main__":
    main()

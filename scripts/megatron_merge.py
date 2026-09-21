# Copyright 2025 the ROLL team and the LlamaFactory team.
#
# This code is modified from the ROLL library.
# https://github.com/alibaba/ROLL/blob/main/mcore_adapter/tools/convert.py
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

import json
import os
from pathlib import Path

import fire
import torch
from mcore_adapter.models.converter.post_converter import convert_checkpoint_to_hf, convert_checkpoint_to_mca
from mcore_adapter.training_args import DistributingParallelArguments
from mcore_adapter.utils import get_logger
from transformers import AutoConfig


logger = get_logger(__name__)


def convert_mca_to_hf(
    checkpoint_path: str,
    output_path: str = "./output",
    bf16: bool = False,
    fp16: bool = False,
    convert_model_max_length: int | None = None,
):
    """Convert megatron checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to the checkpoint to convert
        output_path: Path to save the converted checkpoint
        bf16: Use bfloat16 precision
        fp16: Use float16 precision
        convert_model_max_length: Change the model_max_length in hf config.json
    """
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot be both True.")

    torch_dtype = None
    if bf16:
        torch_dtype = torch.bfloat16
    elif fp16:
        torch_dtype = torch.float16

    convert_checkpoint_to_hf(checkpoint_path, output_path, torch_dtype=torch_dtype)

    if convert_model_max_length is not None:
        config = AutoConfig.from_pretrained(output_path, trust_remote_code=True)
        config.model_max_length = convert_model_max_length
        config.save_pretrained(output_path)


def export_training_checkpoint(config):
    """Convert the final native checkpoint using the resolved training configuration."""
    if isinstance(config, str):
        config = json.loads(config)
    checkpoint = Path(config["output_dir"])
    output = Path(config.get("export_dir") or (str(checkpoint) + "-hf"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"HF export destination is not empty: {output}")
    metadata = checkpoint / "mca_config.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing final Megatron checkpoint metadata: {metadata}")
    trained = json.loads(metadata.read_text())
    max_length = trained.get("max_sequence_length")
    convert_mca_to_hf(
        str(checkpoint),
        str(output),
        bf16=config.get("bf16", False),
        fp16=config.get("fp16", False),
        convert_model_max_length=max_length,
    )
    hf_config = AutoConfig.from_pretrained(output, trust_remote_code=config.get("trust_remote_code", False))
    if trained.get("mrope_yarn_enabled"):
        text_config = getattr(hf_config, "text_config", hf_config)
        text_config.max_position_embeddings = max_length
        hf_config.max_position_embeddings = max_length
        text_config.rope_parameters.update(
            rope_type="yarn",
            factor=trained["yarn_rotary_scaling_factor"],
            original_max_position_embeddings=trained["yarn_original_max_position_embeddings"],
            beta_fast=trained["yarn_beta_fast"],
            beta_slow=trained["yarn_beta_slow"],
            mscale=trained["yarn_mscale"],
            mscale_all_dim=trained["yarn_mscale_all_dim"],
            truncate=trained.get("yarn_correction_range_round_to_int", False),
        )
        hf_config.save_pretrained(output)
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=config.get("trust_remote_code", False))
    if not tokenizer.encode("Training export validation"):
        raise ValueError("Exported tokenizer failed validation.")
    template = checkpoint / "chat_template.jinja"
    if template.exists() and (output / template.name).read_bytes() != template.read_bytes():
        raise ValueError("Exported chat template differs from the trained template.")
    index = output / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        if not weight_map:
            raise ValueError("Empty exported weight index.")
        for filename in set(weight_map.values()):
            with safe_open(str(output / filename), framework="pt", device="cpu") as shard:
                if set(shard.keys()) != {key for key, value in weight_map.items() if value == filename}:
                    raise ValueError(f"Invalid exported shard: {filename}")
    else:
        with safe_open(str(output / "model.safetensors"), framework="pt", device="cpu") as shard:
            if not list(shard.keys()):
                raise ValueError("Empty exported weights.")
    print(f"Validated HF export: {output}", flush=True)


def convert(
    checkpoint_path: str | None = None,
    output_path: str = "./output",
    bf16: bool = False,
    fp16: bool = False,
    convert_model_max_length: int | None = None,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: int | None = None,
    moe_grouped_gemm: bool | None = None,
    training_config: str | dict | None = None,
):
    """Convert checkpoint between MCA and HuggingFace formats.

    Args:
        checkpoint_path: Path to the checkpoint to convert
        output_path: Path to save the converted checkpoint
        bf16: Use bfloat16 precision
        fp16: Use float16 precision
        convert_model_max_length: Change the model_max_length in hf config.json
        training_config: Resolved training config dict or JSON for final HF export.
        tensor_model_parallel_size: Tensor model parallel size
        pipeline_model_parallel_size: Pipeline model parallel size
        expert_model_parallel_size: Expert model parallel size
        virtual_pipeline_model_parallel_size: Virtual pipeline model parallel size
        moe_grouped_gemm: Use grouped gemm for MoE experts. When enabled, expert
            weights are stored in a flattened format (linear_fc1.weight0, weight1, ...)
            rather than per-expert format (local_experts.0.linear_fc1.weight, ...).
            Must match the format used when saving the checkpoint.
    """
    if bf16 and fp16:
        raise ValueError("bf16 and fp16 cannot be both True.")

    if training_config is not None:
        return export_training_checkpoint(training_config)
    if checkpoint_path is None:
        raise ValueError("checkpoint_path or training_config is required.")

    mca_config_path = os.path.join(checkpoint_path, "mca_config.json")
    from_mca = os.path.exists(mca_config_path)

    if not from_mca:
        dist_args = DistributingParallelArguments(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            moe_grouped_gemm=moe_grouped_gemm,
            transformer_impl="transformer_engine",  # hard code here since we default using te for training
        )
        convert_checkpoint_to_mca(
            checkpoint_path,
            output_path,
            dist_args,
            bf16=bf16,
            fp16=fp16,
        )
    else:
        convert_mca_to_hf(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            bf16=bf16,
            fp16=fp16,
            convert_model_max_length=convert_model_max_length,
        )


def main():
    fire.Fire(convert)


if __name__ == "__main__":
    main()

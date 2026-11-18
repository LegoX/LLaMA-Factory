import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("--work-dir", type=Path, required=True)
parser.add_argument("--tokenizer-source", type=Path, required=True)
args = parser.parse_args()
root = args.work_dir.resolve()
modeldir = root / "model"
modeldir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
config = Qwen3_5Config(
    text_config={
        "vocab_size": 248320,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 1024,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 3, 3],
        },
        "use_cache": False,
    },
    vision_config={
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 128,
        "num_position_embeddings": 64,
    },
    image_token_id=248056,
    video_token_id=248057,
    vision_start_token_id=248053,
    tie_word_embeddings=False,
)
model = Qwen3_5ForConditionalGeneration(config)
model.save_pretrained(modeldir)
source = args.tokenizer_source.resolve()
for p in source.iterdir():
    if (
        p.name not in ["config.json", "generation_config.json"]
        and p.suffix in (".json", ".jinja", ".txt")
        and "safetensors" not in p.name
    ):
        shutil.copy2(p, modeldir / p.name)
(root / "data.json").write_text(
    json.dumps([{"instruction": f"What is {i} plus one?", "input": "", "output": str(i + 1)} for i in range(4)])
)
train = {
    "model_name_or_path": str(modeldir),
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "dataset": "tiny",
    "template": "qwen3_6",
    "enable_thinking": False,
    "cutoff_len": 128,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "bf16": True,
    "learning_rate": 1e-4,
    "max_steps": 2,
    "warmup_steps": 0,
    "lr_scheduler_type": "constant",
    "save_strategy": "steps",
    "save_steps": 1,
    "save_total_limit": 2,
    "logging_steps": 1,
    "report_to": "none",
    "output_dir": str(root / "full"),
    "flash_attn": "sdpa",
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "preprocessing_num_workers": 1,
    "dataloader_num_workers": 0,
    "launcher": {
        "profile": "qwen36_fsdp2",
        "device": "npu",
        "nproc_per_node": 2,
        "min_host_available_gib": 1,
        "dataset_info": {"tiny": {"file_name": str(root / "data.json")}},
        "optimizations": {"gdn_segment": 128, "mlp_chunk": 64, "norm_chunk": 64},
    },
}
(root / "train.yaml").write_text(yaml.safe_dump(train, sort_keys=False))
train["output_dir"] = str(root / "resumed")
train["resume_from_checkpoint"] = str(root / "full/checkpoint-1")
(root / "resume.yaml").write_text(yaml.safe_dump(train, sort_keys=False))
print(root)

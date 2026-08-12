# Qwen3.5-35B-A3B SFT Quickstart

This guide walks through a full supervised fine-tune of Qwen3.5-35B-A3B-Base, from
environment setup to launching training, using the public sample dataset published by
SWE-Lego. All paths are relative to the repository root; models and datasets can also be
referenced directly by Hugging Face Hub ID.

## 1. Install the environment

Clone the repository and enter it:

```bash
git clone https://github.com/SWE-Lego/LLaMA-Factory.git
cd LLaMA-Factory
```

Run the installer:

```bash
bash install_env.sh
conda activate lf_v3
```

This installs LLaMA-Factory, PyTorch, DeepSpeed, FlashAttention, Liger Kernel and the
remaining dependencies required for Qwen3.5 training. See
[`install_env.sh`](../install_env.sh) for the pinned versions and why each one matters.

## 2. Prepare the data

A sample dataset is published on Hugging Face under
[`SWE-Lego/samples_for_llama_factory_sft`](https://huggingface.co/datasets/SWE-Lego/samples_for_llama_factory_sft).

Download the sample JSON into the repository's `data` directory:

```bash
curl -L \
  -o data/sft_sample_for_qwen3_5.json \
  https://huggingface.co/datasets/SWE-Lego/samples_for_llama_factory_sft/resolve/main/jierun_glm5_swerebench_oraclesolved_oh_sdk_512_for_qwen3_5.json
```

The data stores multi-turn conversations in a `messages` field. Keep this structure:

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "The user question or task description"},
      {"role": "assistant", "content": "The response the model should learn"}
    ]
  }
]
```

Use `system`, `user` and `assistant` for `role`. The training loss comes primarily from
the `assistant` messages.

## 3. Register the dataset

Add an entry to `data/dataset_info.json`:

```json
{
  "sft_sample_for_qwen3_5": {
    "file_name": "sft_sample_for_qwen3_5.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

To train without downloading anything locally, publish the data to your own Hugging Face
dataset repository and register it with `hf_hub_url` instead:

```json
{
  "my_sft_dataset": {
    "hf_hub_url": "your-org/your-sft-dataset",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

> If your dataset is a single JSON file larger than 2 GiB, split it into several smaller
> files inside a directory and point `file_name` at that directory — PyArrow's 32-bit
> string offsets overflow on larger single files. See
> [the multi-node notes](qwen3_5_moe_sft_multinode_notes.md#1-pyarrow-int32-string-offset-overflow).

## 4. Write the training config

The repository ships full-SFT examples for Qwen3.5-35B-A3B under `examples/train_full/`,
which you can use as a starting point. To create your own, e.g.
`examples/train_full/qwen3_5_35b_a3b_sft_sample.yaml`:

```yaml
### model
model_name_or_path: Qwen/Qwen3.5-35B-A3B-Base
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: sft_sample_for_qwen3_5
dataset_dir: data
template: qwen3_5_nothink
cutoff_len: 131072
rope_scaling: yarn
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3_5_35b_a3b_sft
logging_steps: 1
save_strategy: steps
save_steps: 50
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
weight_decay: 0.01
max_grad_norm: 1.0
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
enable_liger_kernel: true
use_unsloth_gc: true
flash_attn: fa2
```

Tune `cutoff_len`, `per_device_train_batch_size` and `gradient_accumulation_steps` to fit
your GPU memory and sequence lengths. `save_strategy: steps` with a small `save_steps` is
recommended for long runs so that an interrupted job loses minutes rather than hours.

To log to Weights & Biases, set `report_to: wandb` and export your key before launching:

```bash
export WANDB_API_KEY=...   # never commit this
```

## 5. Launch training

Single node, multiple GPUs:

```bash
TRAIN_CONFIG=examples/train_full/qwen3_5_35b_a3b_sft_sample.yaml \
bash run_sft_qwen3_5_35b_a3b_base.sh
```

For multi-node training run the same script on every machine, giving each a distinct
`NODE_RANK`:

```bash
TRAIN_CONFIG=examples/train_full/qwen3_5_35b_a3b_sft_sample.yaml \
NNODES=2 \
NODE_RANK=0 \
bash run_sft_qwen3_5_35b_a3b_base.sh
```

Set `NODE_RANK=1` on the second machine, and increment it for any further nodes. The
launcher derives `gradient_accumulation_steps` from the total GPU count so that the global
batch size stays fixed as you scale nodes up or down.

When training finishes, the weights are written to the `output_dir` from your config.

## Troubleshooting

If you hit dependency, long-context OOM or multi-node collective problems, see
[Qwen3.5-MoE SFT: multi-node and long-context notes](qwen3_5_moe_sft_multinode_notes.md),
which documents the failures this configuration was built to avoid and the reasoning
behind each pinned version and environment variable.

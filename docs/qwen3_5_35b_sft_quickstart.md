# Qwen3.5-35B-A3B SFT 简明使用指引

本文面向社区用户，演示如何从环境安装开始，使用 SWE-Lego 样例数据完成一次 Qwen3.5-35B-A3B-Base 监督微调。文中的文件位置均为仓库内相对位置；模型和数据也可以直接使用 Hugging Face Hub ID。

## 1. 安装环境

克隆仓库并进入仓库根目录：

```bash
git clone https://github.com/SWE-Lego/LLaMA-Factory.git
cd LLaMA-Factory
```

运行环境安装脚本：

```bash
bash install_env.sh
conda activate lf_v3
```

脚本会安装 LLaMA-Factory、PyTorch、DeepSpeed、FlashAttention、Liger Kernel 以及 Qwen3.5 训练所需的相关依赖。

## 2. 准备数据

样例数据位于 Hugging Face：

```text
https://huggingface.co/datasets/SWE-Lego/samples_for_llama_factory_sft/blob/main/jierun_glm5_swerebench_oraclesolved_oh_sdk_512_for_qwen3_5.json
```

下载样例 JSON 到仓库的 `data` 目录：

```bash
curl -L \
  -o data/jierun_glm5_swerebench_oraclesolved_oh_sdk_512_for_qwen3_5.json \
  https://huggingface.co/datasets/SWE-Lego/samples_for_llama_factory_sft/resolve/main/jierun_glm5_swerebench_oraclesolved_oh_sdk_512_for_qwen3_5.json
```

该数据使用 `messages` 字段保存多轮对话，推荐保持如下结构：

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "用户问题或任务描述"},
      {"role": "assistant", "content": "期望模型学习的回答"}
    ]
  }
]
```

其中 `role` 建议使用 `system`、`user`、`assistant`，训练损失主要来自 `assistant` 消息。

## 3. 注册数据

在 `data/dataset_info.json` 中加入一个数据集条目：

```json
{
  "swelego_jierun_glm5_swerebench_oraclesolved_oh_sdk_512_for_qwen3_5": {
    "file_name": "jierun_glm5_swerebench_oraclesolved_oh_sdk_512_for_qwen3_5.json",
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

如果不想下载数据，也可以把数据发布到自己的 Hugging Face dataset repo，然后在 `dataset_info.json` 中使用 `hf_hub_url` 注册：

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

## 4. 配置 `config.yaml`

仓库已在 `examples/train_full` 下提供 Qwen3.5-35B-A3B full SFT 配置示例，可参考：

```text
examples/train_full/qwen3_5_35b_a3b_base_selfmade_traj_selected_gbs64pbs1acc8_lr5e-5_epo3.yaml
```

也可以在同一目录下创建自己的配置，例如 `examples/train_full/qwen3_5_35b_a3b_sft_sample.yaml`：

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
dataset: swelego_jierun_glm5_swerebench_oraclesolved_oh_sdk_512_for_qwen3_5
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

按显存和数据长度调整 `cutoff_len`、`per_device_train_batch_size`、`gradient_accumulation_steps`。如需使用 W&B，将 `report_to` 改为 `wandb` 并提前登录。

## 5. 启动训练

单机多卡训练：

```bash
TRAIN_CONFIG=examples/train_full/qwen3_5_35b_a3b_sft_sample.yaml \
bash run_sft_qwen3_5_35b_a3b_base.sh
```

多机训练时，在每台机器上执行同一脚本，并为每台机器设置不同的 `NODE_RANK`：

```bash
TRAIN_CONFIG=examples/train_full/qwen3_5_35b_a3b_sft_sample.yaml \
NNODES=2 \
NODE_RANK=0 \
bash run_sft_qwen3_5_35b_a3b_base.sh
```

第二台机器将 `NODE_RANK` 改为 `1`；更多机器依次递增。

训练完成后，模型权重会保存到 `output_dir` 指定的位置。

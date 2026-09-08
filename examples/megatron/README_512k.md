# Qwen3.5-35B-A3B-Base 512K YaRN training

## Scope and validation

The [example YAML](qwen3_5_35b_a3b_base_512k_yarn.yaml) is a sanitized single-node **8×H200** full-SFT reference using **LLaMA-Factory → mcore_adapter → Megatron-Core** (`USE_MCA=1`), not Megatron Bridge. It preserves training parameters while replacing machine paths and private dataset/run names.

Optimizer steps have been observed. This guide does **not** certify completion of the full 117-step run, HF export, 512K serving, or evaluation. The 524,288-token cutoff is a maximum, not evidence that all samples have that length or that model quality at that length is established. Preparing this guide does not modify or restart the running job.

## Observed environment

| Component | Version or baseline |
| --- | --- |
| Hardware | One node, 8×H200, approximately 141 GiB device memory per GPU |
| LLaMA-Factory | `0.9.5.dev0`, baseline `ecbc2bb3`, plus local changes below |
| mcore_adapter | `0.10.0.dev0`, adapter baseline `192b1a0`, plus local changes below |
| Megatron-Core | `0.18.2` |
| PyTorch | `2.10.0+cu128` |
| Transformers | `5.6.0` |
| Transformer Engine | `2.18.0` |

This is an environment record, not a complete lockfile. Stock wheels with these labels do not necessarily include the required changes. Host RAM must accommodate optimizer offload and saving; a minimum has not been established here. Do not assume this fits smaller GPUs.

## Local extension checklist

This inventory is **not a bundled patch set**. The adapter is a separate repository: committing this example does not publish its modifications. Before claiming clean-machine reproducibility, review and preserve required changes as versioned commits or reviewed patches, and document how to obtain them. Do not update an environment used by a running job.

| Location | Relevant local changes |
| --- | --- |
| LLaMA-Factory: `src/llamafactory/train/mca/workflow.py` | Avoid cutoff-length padding with EP and variable sequence lengths; align optional loss weights; consult the adapter final-save benchmark flag. |
| Adapter: `mcore_adapter/src/mcore_adapter/training_args.py` | Precision-aware optimizer, cache release, epoch-padding and benchmark flags; explicit FP32 accumulation control. |
| Adapter: `mcore_adapter/src/mcore_adapter/trainer/trainer.py` | Forward optimizer settings; release unused CUDA cache before final gradient synchronization; pad epochs to full steps; terminate/recreate epoch iterators; optional loss weights. |
| Adapter: `mcore_adapter/src/mcore_adapter/models/qwen3_5/config_qwen3_5.py` | Define the mRoPE/YaRN fields used by the YAML. |
| Adapter: `mcore_adapter/src/mcore_adapter/models/qwen3_5/modeling_qwen3_5.py` | Wire YaRN into rotary embeddings and the local long-sequence output path. |
| Adapter: `mcore_adapter/src/mcore_adapter/models/qwen3_vl/rope_utils.py` | Local YaRN frequency interpolation and rotary scaling. |
| Adapter: `mcore_adapter/src/mcore_adapter/models/model_factory.py` | TP-aware fused cross entropy and checkpointed chunked vocabulary-parallel output processing to limit peak logits memory. |

Not every existing diff is required by this dataset: optional loss weighting and benchmark instrumentation are also present. Review actual implementations rather than copying all local changes. Separate HF RoPE/Liger changes, unrelated experiments, and generated files from this documentation change.

## Data and schedule

1. Replace `model_name_or_path` with your base HF model directory.
2. Follow the [dataset format guide](../../data/README.md). Register `long_context_sft` in `dataset_info.json` inside the chosen `dataset_dir`. Data and registration files are not bundled.
3. Perform any benchmark-repository exclusion during data preparation. This YAML does not perform contamination filtering.
4. Choose a fresh `output_dir`, never an active run's directory. Relative paths resolve from the launch directory.

The `qwen3_6` template matches the reference run rather than being inferred from the model name. Verify compatibility with your prompt and supervision format. `packing: false` and `variable_seq_lengths: true` retain variable lengths with a 524,288-token cap.

For **2,481 samples** and effective batch **64**, the local padding sampler gives `ceil(2481 / 64) = 39` steps per epoch: **117 steps represent three padded epochs**. Each epoch has 2,496 sample slots, repeating 15 samples with rotating padding. This is not exactly three unpadded passes.

The YAML retains `max_steps: 117` and `lr_scheduler_kwargs.lr_decay_steps: 117`. Recompute both for different data or batch sizes; 117 is not a generic three-epoch setting. In this exact topology, effective batch is `1 × 64 × 1` (microbatch × accumulation × data parallel size). EP is not an additional independent multiplier on top of TP × PP × CP.

## Training settings

- TP2 / PP2 / CP2 / EP4 / ETP1, sequence parallelism, full activation recomputation.
- BF16 forward/backward and FP32 gradient accumulation/all-reduce buffers. This is not FP8 training, regardless of the trajectory generator's precision.
- Distributed precision-aware optimizer with 75% CPU offload; parameter-gather and gradient-reduce overlap disabled.
- mRoPE with YaRN factor 2, original length 262,144 and maximum length 524,288; these adapter-specific fields require the changes listed above.
- `fine_grained_activation_offloading: false`: the retained `offload_modules` list does not enable activation offload. Optimizer offload is separate.

## Launch after preparing dependencies

Activate the prepared environment. From the repository root, replace the adapter path below with the reviewed patched checkout. Launch only when all eight GPUs are available.

```bash
env USE_MCA=1 NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTHONPATH="/path/to/patched-adapter/mcore_adapter/src${PYTHONPATH:+:$PYTHONPATH}" \
  llamafactory-cli train examples/megatron/qwen3_5_35b_a3b_base_512k_yarn.yaml
```

In tmux, run in the foreground of the designated pane with stdout/stderr attached. Do not use `tee`, shell redirection, or a background launcher. Persist output separately with tmux-native pane capture if needed.

The reference job also supplied `PYTHONWARNINGS=ignore::UserWarning` and `TORCH_CPP_LOG_LEVEL=ERROR` to reduce known warning noise. They are omitted here so diagnostic warnings remain visible. A one-time Dynamo recompilation/fallback warning is not itself evidence of failure. The first step log may be delayed by compilation and 64 accumulated microbatches; inspect actual exceptions and progress before intervening.

## Saving and later export

`save_strategy: "no"` disables periodic checkpoints. The local workflow still calls final `trainer.save_model()` unless `benchmark_skip_final_save` is enabled; this example leaves its patched default at `false`. `save_only_model: true`, `save_hf_model: false`, and `ckpt_format: torch_dist` select final MCore model output, not HF weights or a full optimizer-resumable checkpoint. Logs and trainer metadata may still be written.

There are no intermediate checkpoints or optimizer state for an exact interrupted-run resume. Verify all final model shards and metadata before conversion. Use a separately validated adapter-compatible MCore-to-HF procedure, checking that Qwen3.5 configuration and YaRN fields survive export. No unverified conversion command is provided here.

## Sharing safely

The example uses generic paths and dataset/run names without credentials, data, checkpoints, environment dumps, or private conversion metadata. Review the intended diff before committing; do not stage the entire working tree. Exclude active run configs, unrelated experiments, generated dependencies, logs, and model outputs.

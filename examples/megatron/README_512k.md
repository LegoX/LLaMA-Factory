# Qwen3.5-35B-A3B-Base 512K YaRN training

## Scope and validation

The [example YAML](qwen3_5_35b_a3b_base_512k_yarn.yaml) is a sanitized single-node **8×H200** full-SFT reference using **LLaMA-Factory → mcore_adapter → Megatron-Core** (`USE_MCA=1`), not Megatron Bridge. It preserves training parameters while replacing machine paths and private dataset/run names.

The original reference run reached optimizer steps. That historical observation does **not** certify the current integration, completion of the full 117-step run, HF export, 512K serving, or evaluation. The 524,288-token cutoff is a maximum, not evidence that all samples have that length or that model quality at that length is established. See the [integration guide](../../docs/long_context_sft.md) for all eight feature areas and current validation.

## Observed environment

| Component | Version or baseline |
| --- | --- |
| Hardware | One node, 8×H200, approximately 141 GiB device memory per GPU |
| LLaMA-Factory | Original reference job: `0.9.5.dev0`, baseline `ecbc2bb3`, plus local changes; current integration provenance is recorded in the feature guide |
| mcore_adapter | `0.10.0.dev0`, vendored in-tree at `src/mcore_adapter` from ROLL commit `192b1a01`, plus local changes below |
| Megatron-Core | `0.18.2` |
| PyTorch | `2.10.0+cu128` |
| Transformers | `5.6.0` |
| Transformer Engine | `2.18.0` |
| cuDNN | `9.26.0.51`, installed over the `9.10.2.21` the torch wheel pins; 9.10 fused attention is wrong in THD (packed) layout for head_dim 256 |

The bundled [constraints](512k/constraints.txt) pin the reference Python package versions; local editable packages and private data-processing tools are excluded. The adapter source is tracked in this repository. This does not lock OS packages, drivers, or compiler build dependencies. Use Linux x86-64, Python 3.12, a CUDA 12.8 toolkit (observed nvcc 12.8.93), and a CUDA-12.8-compatible NVIDIA driver. Host RAM must accommodate optimizer offload and saving; a minimum has not been established here. Do not assume this fits smaller GPUs.

## Local extension checklist

The LLaMA-Factory `workflow.py` integration is included in this branch. The adapter is vendored in-tree at `src/mcore_adapter` (Apache-2.0, from `alibaba/ROLL` commit `192b1a01ea61c113b2deb543f7b115783038dff8`); the commit history of that directory records the pristine upstream import followed by the local changes below, so no separate patch bundle or checkout is needed.

| Location | Relevant local changes |
| --- | --- |
| LLaMA-Factory: `src/llamafactory/train/mca/workflow.py` | Avoid cutoff-length padding with EP and variable sequence lengths; align optional loss weights; consult the adapter final-save benchmark flag. |
| Adapter: `src/mcore_adapter/training_args.py` | Precision-aware optimizer, cache release, epoch-padding and benchmark flags; explicit FP32 accumulation control. |
| Adapter: `src/mcore_adapter/trainer/trainer.py` | Forward optimizer settings; release unused CUDA cache before final gradient synchronization; pad epochs to full steps; terminate/recreate epoch iterators; optional loss weights. |
| Adapter: `src/mcore_adapter/models/qwen3_5/config_qwen3_5.py` | Define the mRoPE/YaRN fields used by the YAML. |
| Adapter: `src/mcore_adapter/models/qwen3_5/modeling_qwen3_5.py` | Wire YaRN into rotary embeddings and the local long-sequence output path. |
| Adapter: `src/mcore_adapter/models/qwen3_vl/rope_utils.py` | Local YaRN frequency interpolation and rotary scaling. |
| Adapter: `src/mcore_adapter/models/model_factory.py` | TP-aware fused cross entropy and checkpointed chunked vocabulary-parallel output processing to limit peak logits memory. |
| Adapter: `src/mcore_adapter/models/converter/model_converter.py` | Ignore auxiliary MTP weights when MTP training is disabled. |
| Adapter: `src/mcore_adapter/models/sequence_packing_mixin.py`, `trainer.py`, `models/qwen3_5/modeling_qwen3_5.py` | Flat THD layout for `neat_packing` with context parallelism, `LF_PACK_SUBSEQ_ALIGN` boundary alignment, mRoPE position alignment for packed rows. |
| Adapter: `src/mcore_adapter/trainer/trainer.py` | `EpochShuffledSampler`: seeded per-epoch permutation with exact global-step drop_last across data-parallel ranks, enabled by `MCA_EPOCH_SHUFFLE_SEED`. |
| Adapter: `src/mcore_adapter/trainer/trainer.py` | Scale the per-token loss by `cp_size` under context parallelism: every CP rank reports the CP-summed token count and Megatron divides gradients by the DP×CP-reduced count, so without the factor gradients were `1/cp_size` of the correct value (reported `grad_norm` halved at CP2). |

The integration retains the long-context implementation and adds supervision controls, scoring, correctness fixes, diagnostics and export. In particular, [loss weights now scale the numerator over a valid-token denominator](../../docs/long_context_sft.md#template-and-supervision). Optional features need not be enabled for this YAML. Private datasets and per-run configurations are not included.

## Data and schedule

1. Replace `model_name_or_path` with your base HF model directory.
2. Follow the [dataset format guide](../../data/README.md). Register `long_context_sft` in `dataset_info.json` inside the chosen `dataset_dir`. Data and registration files are not bundled.
3. Perform any benchmark-repository exclusion during data preparation. This YAML does not perform contamination filtering.
4. Choose a fresh `output_dir`, never an active run's directory. The common launcher resolves relative paths from the repository root.

The `qwen3_6` template matches the reference run rather than being inferred from the model name. Verify compatibility with your prompt and supervision format. `packing: false` and `variable_seq_lengths: true` retain variable lengths with a 524,288-token cap.

For **2,481 samples** and effective batch **64**, the local padding sampler gives `ceil(2481 / 64) = 39` steps per epoch: **117 steps represent three padded epochs**. Each epoch has 2,496 sample slots, repeating 15 samples with rotating padding. This is not exactly three unpadded passes.

The YAML retains `max_steps: 117` and `lr_scheduler_kwargs.lr_decay_steps: 117`. Recompute both for different data or batch sizes; 117 is not a generic three-epoch setting. In this exact topology, effective batch is `1 × 64 × 1` (microbatch × accumulation × data parallel size). EP is not an additional independent multiplier on top of TP × PP × CP.

## Training settings

- TP2 / PP2 / CP2 / EP4 / ETP1, sequence parallelism, full activation recomputation.
- BF16 forward/backward and FP32 gradient accumulation/all-reduce buffers. This is not FP8 training, regardless of the trajectory generator's precision.
- Distributed precision-aware optimizer with 75% CPU offload; parameter-gather and gradient-reduce overlap disabled.
- mRoPE with YaRN factor 2, original length 262,144 and maximum length 524,288; these adapter-specific fields require the changes listed above.
- `fine_grained_activation_offloading: false`: the retained `offload_modules` list does not enable activation offload. Optimizer offload is separate.

## Install into a separate environment

Use a checkout of this integration, which includes `src/mcore_adapter` and the common launcher. It is based on `LegoX/LLaMA-Factory`'s `conghao/feature`; the upstream public checkout does not contain these extensions. Install system prerequisites first: conda, Git, a C++ compiler/build tools, and CUDA 12.8 with `nvcc`. Network access to GitHub, PyPI, and the PyTorch wheel index is required. CUDA extensions may compile from source; allow disk space and time for compilation.

From this repository root:

```bash
export CUDA_HOME=/path/to/cuda-12.8
bash examples/megatron/512k/install.sh /work/envs/lf
```

The target path must be absolute and must not already exist. The script creates a conda env, installs the pinned PyTorch CUDA 12.8 build and runtime dependencies, installs this repository (which ships `mcore_adapter`) in editable mode, overrides cuDNN to 9.26.0.51, and runs `pip check` (the only expected complaint is the cuDNN pin of the torch wheel). It does not start training. It deliberately refuses to update an existing environment. If installation fails, inspect the error rather than rerunning against an active environment. The old generic Megatron Dockerfile targets different versions and is not this recipe.

An existing environment with torch's cuDNN `9.10.2.21` can also pass preflight
when both `NVTE_FLASH_ATTN=1` and `NVTE_FUSED_ATTN=0` are explicitly set. This
disables the cuDNN attention backend affected by the THD issue. All other pinned
versions remain required. Without these settings, preflight requires cuDNN
`9.26.0.51`. GPU execution still needs validation on the target hardware.

For interactive use, `source examples/megatron/512k/activate_shared.sh /path/to/env` exports the same variables as `run.sh`, using this checkout. See [the unified feature guide](../../docs/long_context_sft.md) for Qwen3.8, supervision weights, scoring, diagnostics and export.

## Prepare data and launch

The earlier data section describes how to register your own dataset. For example, an Alpaca-format JSON array of records with `instruction`, `input`, and `output` can be registered inside the selected dataset directory with:

```json
{
  "long_context_sft": {
    "file_name": "train.json",
    "formatting": "alpaca",
    "columns": {"prompt": "instruction", "query": "input", "response": "output"}
  }
}
```

This is a schema example, not the private reference dataset. Use the actual matching data format and benchmark filtering for your experiment. Copy the supplied YAML to a new file, replace its model/data paths, and set a fresh output directory; retain the reference parameters when reproducing that run. For a one-step acceptance run on free GPUs, use a separate config with `max_steps: 1`, `lr_scheduler_kwargs: {lr_decay_steps: 1}`, and a new output directory. Use representative long samples; a short synthetic sample does not validate 512K memory behavior.

Once all eight GPUs are available, launch from the repository root:

```bash
bash examples/megatron/512k/run.sh /work/envs/lf /work/configs/train-512k.yaml
```

Multi-node: export `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` (and the NCCL interface variables for the fabric) on every node and run the same command on each; the launcher passes them to torchrun.

The launcher checks pinned core versions, in-tree source identity, YAML fields, local model/data registration, and the output directory (an existing directory requires `resume_from_checkpoint`). It sets `USE_MCA=1`, defaults `NPROC_PER_NODE` to eight, selects this checkout and packaged NVIDIA library paths, and defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It then runs the foreground training CLI, which converts the final native checkpoint after the workers exit successfully. Preflight hides GPUs in its own child process before importing third-party libraries and never loads model weights; training preserves the caller's `CUDA_VISIBLE_DEVICES`.

On multi-node runs, node 0 checks whether the output directory already exists;
other nodes may reach preflight after node 0 has created it. Every node still
checks its model and data paths.

Append `key=value` arguments to override YAML settings. Prefix the command with `DRY_RUN=1` for topology/config checks without CUDA dependency imports, or `PREFLIGHT_ONLY=1` for full environment/config/input checks without training. Single-node launches clear inherited distributed rank settings; multi-node launches use the environment above. See the [launch guide](../../docs/long_context_sft.md#launch-and-environment) for examples.

In tmux, run in the foreground of the designated pane with stdout/stderr attached. Do not add `tee`, output redirection, or background launchers. Persist output separately with tmux-native pane capture if needed.

The reference job also supplied `PYTHONWARNINGS=ignore::UserWarning` and `TORCH_CPP_LOG_LEVEL=ERROR` to reduce known warning noise. They are omitted here so diagnostic warnings remain visible. A one-time Dynamo recompilation/fallback warning is not itself evidence of failure. The first step log may be delayed by compilation and 64 accumulated microbatches; inspect actual exceptions and progress before intervening.

## Verification boundary

The in-tree implementation has CPU regression coverage and a CPU-only import/config preflight. The current integration's test results, wheel build, and remaining GPU checks are recorded in the [validation section](../../docs/long_context_sft.md#validation).

The CPU regression tests cover in-tree source layout, label/loss-weight shifting, EP/variable-length padding, and the epoch-padding sampler:

```bash
CUDA_VISIBLE_DEVICES='' WANDB_DISABLED=true PYTHONPATH=src /work/envs/lf/bin/python -B -m unittest discover -s tests/train -p test_mca_512k_bundle.py -v
```

These checks use existing installed dependencies with this source checkout. An isolated wheel build succeeded and includes the adapter, scoring and export modules. A fresh CUDA environment installation and an eight-GPU optimizer step have not been executed for this integration. The complete training/export/evaluation chain still needs GPU acceptance testing.

## Saving and later export

`save_strategy: "no"` disables periodic checkpoints. The local workflow still calls final `trainer.save_model()` unless `benchmark_skip_final_save` is enabled; this example leaves its default at `false`. `save_only_model: true`, `save_hf_model: false`, and `ckpt_format: torch_dist` select final MCore model output, not HF weights or a full optimizer-resumable checkpoint. Logs and trainer metadata may still be written.

This example saves no intermediate checkpoints or optimizer state for an exact interrupted-run resume. Verify all final model shards and metadata before manual conversion. The training CLI automatically converts the final native checkpoint using the packaged `llamafactory.model.model_utils.megatron_export` module, to `export_dir` or `output_dir + "-hf"`. `scripts/megatron_merge.py` is a compatibility shim to that implementation. Conversion preserves trained MRoPE/YaRN fields and validates the exported tokenizer, chat template and weight shards. Direct HF saving, skipped final saves and score-only runs bypass automatic conversion. See the [export guide](../../docs/long_context_sft.md#hf-export) for manual conversion and legacy metadata repair.

Keep `save_hf_model: false` when `mrope_yarn_enabled: true`. Direct HF saving is rejected for this combination because it skips YaRN metadata finalization and legacy weight compensation.

## Sharing safely

The example uses generic paths and dataset/run names without credentials, data, checkpoints, environment dumps, or private conversion metadata. Review the intended diff before committing; do not stage the entire working tree. Exclude active run configs, unrelated experiments, generated dependencies, logs, and model outputs. The in-tree workflow and adapter are the implementation required by this recipe.

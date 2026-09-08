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

The bundled [constraints](512k/constraints.txt) pin the observed installed Python package versions; local editable packages and private data-processing tools are excluded. The adapter source is pinned separately below. This does not lock OS packages, drivers, or compiler build dependencies. Use Linux x86-64, Python 3.12, a CUDA 12.8 toolkit (observed nvcc 12.8.93), and a CUDA-12.8-compatible NVIDIA driver. Host RAM must accommodate optimizer offload and saving; a minimum has not been established here. Do not assume this fits smaller GPUs.

## Local extension checklist

The LLaMA-Factory `workflow.py` integration is included in this branch. The six-file adapter implementation is included as [mcore-adapter.patch](512k/mcore-adapter.patch), against official ROLL commit `192b1a01ea61c113b2deb543f7b115783038dff8`. The [manifest](512k/manifest.json) records before/after file checksums and the LLaMA-Factory integration checksum. Applying this bundle reconstructs the adapter source used by the reference job; access to its original machine is not required. Do not patch a checkout or environment being used by a running job.

| Location | Relevant local changes |
| --- | --- |
| LLaMA-Factory: `src/llamafactory/train/mca/workflow.py` | Avoid cutoff-length padding with EP and variable sequence lengths; align optional loss weights; consult the adapter final-save benchmark flag. |
| Adapter: `mcore_adapter/src/mcore_adapter/training_args.py` | Precision-aware optimizer, cache release, epoch-padding and benchmark flags; explicit FP32 accumulation control. |
| Adapter: `mcore_adapter/src/mcore_adapter/trainer/trainer.py` | Forward optimizer settings; release unused CUDA cache before final gradient synchronization; pad epochs to full steps; terminate/recreate epoch iterators; optional loss weights. |
| Adapter: `mcore_adapter/src/mcore_adapter/models/qwen3_5/config_qwen3_5.py` | Define the mRoPE/YaRN fields used by the YAML. |
| Adapter: `mcore_adapter/src/mcore_adapter/models/qwen3_5/modeling_qwen3_5.py` | Wire YaRN into rotary embeddings and the local long-sequence output path. |
| Adapter: `mcore_adapter/src/mcore_adapter/models/qwen3_vl/rope_utils.py` | Local YaRN frequency interpolation and rotary scaling. |
| Adapter: `mcore_adapter/src/mcore_adapter/models/model_factory.py` | TP-aware fused cross entropy and checkpointed chunked vocabulary-parallel output processing to limit peak logits memory. |

The bundle preserves the exact six-file reference implementation, including optional loss weighting and benchmark instrumentation; those optional features need not be enabled for this YAML. It does not include unrelated HF RoPE/Liger edits, private datasets, or other experimental configurations.

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

## Install into a separate environment

Use a fresh checkout of the private `LegoX/LLaMA-Factory` repository on `conghao/feature` containing this bundle, not an upstream public checkout. Install system prerequisites first: Python 3.12 with venv support, Git, a C++ compiler/build tools, and CUDA 12.8 with `nvcc`. Network access to GitHub, PyPI, and the PyTorch wheel index is required. CUDA extensions may compile from source; allow disk space and time for compilation.

From this repository root:

```bash
export CUDA_HOME=/path/to/cuda-12.8
bash examples/megatron/512k/install.sh /work/venvs/qwen35-512k /work/src/roll-qwen35-512k
```

Both target paths must be absolute and must not already exist. The script creates an isolated venv, installs the pinned PyTorch CUDA 12.8 build and runtime dependencies, checks out the fixed official ROLL commit, applies and verifies the included patch, installs both projects, and runs `pip check`. It does not start training. It deliberately refuses to update an existing environment. If installation fails, inspect the error rather than rerunning against an active environment. The old generic Megatron Dockerfile targets different versions and is not this recipe.

For an already prepared **separate** environment and clean adapter checkout, the patch tool can also be used directly:

```bash
python examples/megatron/512k/prepare_adapter.py /work/src/roll-qwen35-512k --apply
```

Without `--apply` it is read-only verification. It refuses a different baseline, partial patches, unrelated tracked edits, or checksum mismatches. Reapplying the exact bundle is a verified no-op.

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
bash examples/megatron/512k/run.sh /work/venvs/qwen35-512k /work/src/roll-qwen35-512k /work/configs/train-512k.yaml
```

The launcher checks source hashes, pinned core versions, all training/model YAML fields, local model/data registration, and that the output directory is new. It sets `USE_MCA=1`, eight processes, the intended source paths, packaged NVIDIA library paths, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as in the reference environment. Then it replaces itself with the foreground training process. The preflight deliberately hides GPUs only in its own child process and never loads model weights; training retains all eight GPUs.

In tmux, run in the foreground of the designated pane with stdout/stderr attached. Do not add `tee`, output redirection, or background launchers. Persist output separately with tmux-native pane capture if needed.

The reference job also supplied `PYTHONWARNINGS=ignore::UserWarning` and `TORCH_CPP_LOG_LEVEL=ERROR` to reduce known warning noise. They are omitted here so diagnostic warnings remain visible. A one-time Dynamo recompilation/fallback warning is not itself evidence of failure. The first step log may be delayed by compilation and 64 accumulated microbatches; inspect actual exceptions and progress before intervening.

## Verification boundary

The bundle is checked by reconstructing separate clean source checkouts, applying the packaged patch, comparing all six patched files byte-for-byte with the reference implementation, and running CPU tests and a CPU-only import/config preflight. The reference installed environment also passes `pip check`; installed Megatron-Core, Transformer Engine Torch, FLA, and FLA Core Python sources were checked against their distribution RECORD hashes without finding additional local edits.

The CPU regression tests cover the bundle hashes, label/loss-weight shifting, EP/variable-length padding, and the epoch-padding sampler:

```bash
CUDA_VISIBLE_DEVICES='' MCA_512K_ADAPTER=/work/src/roll-qwen35-512k \
  /work/venvs/qwen35-512k/bin/python -B -m unittest discover -s tests/train -p test_mca_512k_bundle.py -v
```

These checks use existing installed dependencies with clean **source** checkouts. A fresh package installation/build and an isolated eight-GPU optimizer step have not been executed as part of packaging, because the training environment and GPUs must remain untouched. The files needed to reconstruct the implementation are now included; this is not a claim that a fresh-machine GPU acceptance test or the complete training/export/evaluation chain has passed.

## Saving and later export

`save_strategy: "no"` disables periodic checkpoints. The local workflow still calls final `trainer.save_model()` unless `benchmark_skip_final_save` is enabled; this example leaves its patched default at `false`. `save_only_model: true`, `save_hf_model: false`, and `ckpt_format: torch_dist` select final MCore model output, not HF weights or a full optimizer-resumable checkpoint. Logs and trainer metadata may still be written.

There are no intermediate checkpoints or optimizer state for an exact interrupted-run resume. Verify all final model shards and metadata before conversion. Use a separately validated adapter-compatible MCore-to-HF procedure, checking that Qwen3.5 configuration and YaRN fields survive export. No unverified conversion command is provided here.

## Sharing safely

The example uses generic paths and dataset/run names without credentials, data, checkpoints, environment dumps, or private conversion metadata. Review the intended diff before committing; do not stage the entire working tree. Exclude active run configs, unrelated experiments, generated dependencies, logs, and model outputs. The intentionally included workflow source and adapter patch are the implementation required by this recipe.

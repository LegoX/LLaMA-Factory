# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Code style (auto-fix)
make style

# Code quality check (no modifications)
make quality

# Run all tests
make test

# Run a single test file
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/path/to/test_file.py

# Run tests matching a pattern
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ -k "test_name"

# License header check
make license

# Build package
make build
```

The project uses `uv` as the preferred package manager. Commands automatically use `uv run` / `uvx` if `uv` is available.

## Architecture

LlamaFactory has two parallel architectures controlled by the `USE_V1` environment variable:

- **v0 (default):** `api, webui > chat, eval, train > data, model > hparams > extras`
- **v1 (experimental, `USE_V1=1`):** `trainers > core > accelerator, plugins, config > utils`

Most active development happens in v0. The v1 architecture lives in `src/llamafactory/v1/`.

### Entry Points

CLI entry point is `llamafactory-cli` / `lmf` → `src/llamafactory/cli.py:main()`, which dispatches to `launcher.py` based on `USE_V1`.

Available subcommands: `train`, `chat`, `api`, `export`, `webchat`, `webui`, `env`, `version`, `help`.

### Training Flow (v0)

```
run_exp() [tuner.py]
  → read_args() → parse YAML/JSON config
  → get_train_args() → produces typed argument dataclasses
  → routes to: run_sft / run_dpo / run_ppo / run_rm / run_pt / run_kto
  → optional: export_model()
```

Training is invoked with a YAML config: `llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`

### Configuration System

All training parameters are YAML/JSON config files. Argument parsing in `src/llamafactory/hparams/parser.py` produces four typed dataclasses:
- `ModelArguments` — model/tokenizer selection, quantization
- `DataArguments` — datasets, templates, preprocessing
- `FinetuningArguments` — LoRA rank/target, training method (sft/dpo/ppo/rm/pt/kto)
- `TrainingArguments` — extends HuggingFace's `TrainingArguments`

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/llamafactory/model/loader.py` | Loads model + tokenizer; applies quantization, LoRA, patches |
| `src/llamafactory/model/patcher.py` | Model-specific compatibility patches |
| `src/llamafactory/data/template.py` | Prompt templates; `TEMPLATES` dict maps model family → format |
| `src/llamafactory/data/mm_plugin.py` | Multi-modal (image/video/audio) data handling |
| `src/llamafactory/data/processor/` | Per-stage data processors (supervised, pairwise, pretrain, etc.) |
| `src/llamafactory/train/sft/` | SFT trainer; other stages follow same structure |
| `src/llamafactory/chat/` | Inference engines: `hf_engine`, `vllm_engine`, `sglang_engine`, `kt_engine` |
| `src/llamafactory/extras/constants.py` | Enums and constants used across the project |

### Adding Support for a New Model

1. Add a prompt template to `src/llamafactory/data/template.py` in the `TEMPLATES` dict
2. Add any necessary model patches in `src/llamafactory/model/patcher.py`
3. Add multi-modal support in `src/llamafactory/data/mm_plugin.py` if needed

### Distributed Training

Multi-GPU automatically uses `torchrun`. Additional backends:
- **Ray:** Optional Ray cluster support
- **HyperParallel FSDP2:** `src/llamafactory/train/hyper_parallel/`
- **Megatron-core:** `src/llamafactory/train/mca/`

### Testing

- `tests/` — v0 tests; `tests_v1/` — v1 tests
- Most training tests require GPU hardware
- pytest markers: `@pytest.mark.slow`, `@pytest.mark.runs_on(['cuda'])`
- Always set `WANDB_DISABLED=true` when running tests

### Code Style

- Ruff for linting and formatting (line length 119, Google-style docstrings)
- Python 3.11+ syntax
- Double quotes for strings
- All new files must include Apache 2.0 license header (checked by `make license`)

## Lego-X fork (branch `conghao/feature`)

This fork adds Megatron-Core long-context SFT on top of upstream; the upstream sections above still apply. `AGENTS.md` and `CLAUDE.md` both point at this file.

- **`src/mcore_adapter/`** is vendored from `alibaba/ROLL@192b1a01` (Apache-2.0) and shipped in the wheel together with `src/llamafactory`. Local changes are separate commits on top of the pristine import: 512K YaRN/mRoPE, precision-aware optimizer, chunked cross-entropy, MTP loading guard, flat THD packing with context parallelism, `EpochShuffledSampler` (`MCA_EPOCH_SHUFFLE_SEED`), and the `cp_size` loss scaling (without it gradients were `1/cp_size` under context parallelism). Do not `pip install mcore-adapter`: the PyPI package of that name is unrelated.
- **`examples/megatron/512k/`** is the only launch path: `run.sh <env-prefix> <yaml>` (runs `preflight.py`, then `USE_MCA=1 llamafactory-cli train`), `install.sh <new-conda-prefix>` builds the env from `constraints.txt` with the cuDNN 9.26 override, `activate_shared.sh` for interactive use. Read `examples/megatron/README_512k.md` before changing anything here.
- **Configs**: the repository keeps sanitized examples only (`qwen3_5_35b_a3b_base_512k_yarn.yaml`, `qwen3_6_27b_256k_4node.yaml`). Per-run YAMLs live with their outputs in `saves/<run>/` (a gitignored symlink into `../artifacts/LLaMA-Factory/saves`). `preflight.py` refuses an existing `output_dir`.
- **Multi-node**: the launcher reads `MASTER_ADDR`/`MASTER_PORT`/`NODE_RANK` from the environment and needs `NNODES`, `NPROC_PER_NODE`. With `MCA_EPOCH_SHUFFLE_SEED` the sampler multiple is `per_device_bs × grad_accum × DP`.
- **Numerics**: cuDNN 9.10 fused attention is wrong for THD layout at head_dim 256, so the env pins cuDNN 9.26. Reusing a tokenized cache requires the same `template` and `LF_PACK_SUBSEQ_ALIGN` it was built with.
- Tests for the bundle: `tests/train/test_mca_512k_bundle.py`.

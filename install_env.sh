#!/usr/bin/env bash
#
# install_env.sh — one-shot environment installer for Qwen3.5-35B-A3B-Base SFT.
#
# It creates (or reuses) a conda environment and installs the exact dependency
# versions this fork's training scripts were validated against. The rationale for
# each pinned version is documented in docs/qwen3_5_moe_sft_multinode_notes.md.
#
# What it handles:
#   - torch 2.10.0 + cu128        stable multi-node collectives
#   - flash-linear-attention      required by Qwen3.5 packing-seq forwarding
#   - tilelang                    correct gated_delta_rule backward on Hopper + Triton>=3.4
#   - transformers FA2 s_aux      one-line site-packages patch, reapplied idempotently
#   - liger-kernel dispatch       already patched in this repo's source, picked up by `pip install -e .`
#
# Usage:
#   bash install_env.sh                 # create/reuse conda env "lf_v3"
#   ENV_NAME=my_env bash install_env.sh # custom environment name
#
# To install through a PyPI mirror, export PYPI_INDEX_URL=<mirror url>.
#
# Idempotent: safe to re-run; installed packages and applied patches are skipped.
set -euo pipefail

# ----------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------
# Repository root: defaults to the directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-lf_v3}"
PY_VERSION="${PY_VERSION:-3.12}"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
# Empty means "use pip's configured default index". Set it to a mirror if needed.
PYPI_INDEX_URL="${PYPI_INDEX_URL:-}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.6.0}"
FLA_VERSION="${FLA_VERSION:-0.5.0}"
FSSPEC_VERSION="${FSSPEC_VERSION:-2025.3.0}"

PYPI_ARGS=()
[ -n "$PYPI_INDEX_URL" ] && PYPI_ARGS=(--index-url "$PYPI_INDEX_URL")

log()  { printf '\033[1;32m[install_env]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install_env]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[install_env] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# 1. Locate and initialize conda
# ----------------------------------------------------------------------------
log "Locating conda ..."
# Prefer conda on PATH; otherwise honour CONDA_HOME, then fall back to the usual roots.
if ! command -v conda >/dev/null 2>&1; then
    for guess in "${CONDA_HOME:-}" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [ -n "$guess" ] && [ -x "$guess/bin/conda" ]; then
            export PATH="$guess/bin:$PATH"
            break
        fi
    done
fi
command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda and run 'conda init', or set CONDA_HOME."

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
log "conda base: $CONDA_BASE"

# ----------------------------------------------------------------------------
# 2. Create or reuse the environment
# ----------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Environment '$ENV_NAME' already exists, reusing it."
else
    log "Creating environment '$ENV_NAME' (python=$PY_VERSION) ..."
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

conda activate "$ENV_NAME"
PYBIN="$(command -v python)"
PIP="$PYBIN -m pip"
log "Using python: $PYBIN"
[ -d "$REPO_DIR" ] || die "repository directory not found: $REPO_DIR"
cd "$REPO_DIR"

# ----------------------------------------------------------------------------
# 3. PyTorch 2.10.0 + cu128
#    Earlier 2.8.x releases hang on multi-node _REDUCE_SCATTER_BASE in this setup.
# ----------------------------------------------------------------------------
log "Installing PyTorch $TORCH_VERSION (cu128) ..."
$PIP install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX_URL"

# ----------------------------------------------------------------------------
# 4. LlamaFactory itself plus training extras
#    The liger dispatch fix lives in this repo's source, so the editable install
#    picks it up automatically.
# ----------------------------------------------------------------------------
log "Installing LlamaFactory (editable) ..."
$PIP install "${PYPI_ARGS[@]}" -e .

log "Installing requirements (metrics / deepspeed / liger-kernel) ..."
$PIP install "${PYPI_ARGS[@]}" -r requirements/metrics.txt
$PIP install "${PYPI_ARGS[@]}" -r requirements/deepspeed.txt
$PIP install "${PYPI_ARGS[@]}" -r requirements/liger-kernel.txt

# ----------------------------------------------------------------------------
# 5. flash-attn (FA2)
# ----------------------------------------------------------------------------
log "Installing flash-attn (--no-build-isolation; compilation is slow) ..."
$PIP install \
    "${PYPI_ARGS[@]}" \
    flash-attn \
    --no-build-isolation

# ----------------------------------------------------------------------------
# 6. Pin the Qwen3.5 linear-attention stack
#    LlamaFactory already pulls these in. --no-deps keeps pip from replacing the
#    CUDA 12.8 PyTorch build with a different CUDA variant from PyPI, and stops
#    transformers from drifting off the validated version.
# ----------------------------------------------------------------------------
log "Pinning transformers / flash-linear-attention / fsspec ..."
$PIP install \
    "${PYPI_ARGS[@]}" \
    --no-deps \
    --upgrade \
    "flash-linear-attention==$FLA_VERSION" \
    "fla-core==$FLA_VERSION" \
    "transformers==$TRANSFORMERS_VERSION" \
    "fsspec==$FSSPEC_VERSION"

# ----------------------------------------------------------------------------
# 7. tilelang — required for a correct gated_delta_rule backward on Hopper
#    when Triton >= 3.4 is in use.
# ----------------------------------------------------------------------------
log "Installing tilelang ..."
$PIP install "${PYPI_ARGS[@]}" tilelang

# ----------------------------------------------------------------------------
# 8. Experiment tracking
# ----------------------------------------------------------------------------
log "Installing wandb ..."
$PIP install "${PYPI_ARGS[@]}" wandb

# ----------------------------------------------------------------------------
# 9. Patch the transformers FA2 s_aux=None dereference
#    transformers < 5.7 calls s_aux.to(...) unconditionally in flash_attention.py.
#    During the vision-tower forward s_aux is None, producing
#    "'NoneType' object has no attribute 'to'". The file lives in site-packages, so
#    reinstalling transformers drops the patch; it is reapplied on every run.
# ----------------------------------------------------------------------------
log "Applying transformers FA2 s_aux=None guard ..."
"$PYBIN" - <<'PYEOF'
import os
import transformers

fa = os.path.join(os.path.dirname(transformers.__file__),
                  "integrations", "flash_attention.py")
if not os.path.exists(fa):
    print(f"[install_env]   skipped: {fa} not found")
    raise SystemExit(0)

src = open(fa, encoding="utf-8").read()
buggy = "s_aux=s_aux.to(query.dtype),"
fixed = "s_aux=s_aux.to(query.dtype) if s_aux is not None else None,"

if fixed in src:
    print(f"[install_env]   already fixed ({transformers.__version__}), nothing to do.")
elif buggy in src:
    open(fa, "w", encoding="utf-8").write(src.replace(buggy, fixed, 1))
    print(f"[install_env]   patched -> {fa}")
else:
    # Upstream >= 5.7 fixed this, or the code was restructured. Do not force a
    # rewrite; ask for a manual check instead.
    print(f"[install_env]   target line not found; upstream may already have fixed it"
          f" ({transformers.__version__}). If training reports \"s_aux NoneType.to\","
          " inspect flash_attention.py manually.")
PYEOF

# ----------------------------------------------------------------------------
# 10. Verification
# ----------------------------------------------------------------------------
log "Checking Python dependency consistency ..."
$PIP check

log "Running environment self-check ..."
EXPECTED_TORCH_VERSION="$TORCH_VERSION" \
EXPECTED_TRANSFORMERS_VERSION="$TRANSFORMERS_VERSION" \
EXPECTED_FLA_VERSION="$FLA_VERSION" \
EXPECTED_FSSPEC_VERSION="$FSSPEC_VERSION" \
"$PYBIN" - <<'PYEOF'
import importlib
import os

import torch


def version(module):
    imported = importlib.import_module(module)
    return getattr(imported, "__version__", "ok")


modules = [
    ("transformers", "transformers"),
    ("llamafactory", "llamafactory"),
    ("flash_attn", "flash_attn"),
    ("fla (flash-linear-attention)", "fla"),
    ("fsspec", "fsspec"),
    ("liger_kernel", "liger_kernel"),
    ("tilelang", "tilelang"),
    ("deepspeed", "deepspeed"),
    ("wandb", "wandb"),
]
versions = {"torch": torch.__version__}

print(f"  torch                  : {torch.__version__}")
print(f"  torch.cuda.is_available: {torch.cuda.is_available()}")
try:
    print(f"  bundled NCCL           : {torch.cuda.nccl.version()}")
except Exception:
    pass
for label, module in modules:
    versions[module] = version(module)
    print(f"  {label:<23}: {versions[module]}")

expected = {
    "torch": os.environ["EXPECTED_TORCH_VERSION"],
    "transformers": os.environ["EXPECTED_TRANSFORMERS_VERSION"],
    "fla": os.environ["EXPECTED_FLA_VERSION"],
    "fsspec": os.environ["EXPECTED_FSSPEC_VERSION"],
}
actual_torch = versions["torch"].split("+", 1)[0]
if actual_torch != expected["torch"]:
    raise RuntimeError(f"torch version drift: expected {expected['torch']}, got {versions['torch']}")
for module in ("transformers", "fla", "fsspec"):
    if versions[module] != expected[module]:
        raise RuntimeError(
            f"{module} version drift: expected {expected[module]}, got {versions[module]}"
        )

# Import the parser path that runs LLaMA-Factory's own dependency version checks.
importlib.import_module("llamafactory.hparams")
print("  LLaMA-Factory dependency check: OK")
PYEOF

log "Done. Environment '$ENV_NAME' is ready."
cat <<EOF

Next steps:
  conda activate $ENV_NAME
  cd $REPO_DIR

  # Export your own W&B key first (never commit it)
  export WANDB_API_KEY=...

  # Single node, 8 GPUs
  bash run_sft_qwen3_5_35b_a3b_base.sh

  # Multi-node, e.g. 4 nodes: run once per node with NODE_RANK=0..3
  NNODES=4 NODE_RANK=0 bash run_sft_qwen3_5_35b_a3b_base.sh
EOF

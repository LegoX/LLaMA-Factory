#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: bash install.sh /absolute/new-conda-env-prefix\n'
  exit 2
fi
env_path=$1
if [[ "$env_path" != /* || -e "$env_path" || -L "$env_path" ]]; then
  printf 'Refusing existing or non-absolute target: %s\n' "$env_path"
  exit 2
fi
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$bundle_dir/../../.." && pwd)
: "${CUDA_HOME:?Set CUDA_HOME to a CUDA 12.8 toolkit with nvcc before installation}"
"$CUDA_HOME/bin/nvcc" --version
conda create -y -q -p "$env_path" python=3.12
python_bin="$env_path/bin/python"
export PATH="$env_path/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS="${MAX_JOBS:-4}"
"$python_bin" -m pip install -c "$bundle_dir/constraints.txt" pip setuptools wheel packaging ninja psutil
"$python_bin" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128
"$python_bin" -m pip install --no-build-isolation -c "$bundle_dir/constraints.txt" -r "$bundle_dir/requirements.txt"
"$python_bin" -m pip install -c "$bundle_dir/constraints.txt" -e "$repo_dir"
# cuDNN override: the torch wheel pins 9.10.2.21, whose THD fused attention is wrong for head_dim 256 (packing).
"$python_bin" -m pip install --no-deps nvidia-cudnn-cu12==9.26.0.51
"$python_bin" -m pip check | grep -v 'nvidia-cudnn-cu12' || true
"$python_bin" -c 'import torch, mcore_adapter, llamafactory; print("cudnn", torch.backends.cudnn.version())'
printf 'Installation completed. Run preflight before training; see README_512k.md.\n'

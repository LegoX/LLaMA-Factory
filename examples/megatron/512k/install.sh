#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: bash install.sh /absolute/new-venv /absolute/new-adapter-checkout\n'
  exit 2
fi
venv_path=$1
adapter_path=$2
for target in "$venv_path" "$adapter_path"; do
  if [[ "$target" != /* || -e "$target" || -L "$target" ]]; then
    printf 'Refusing existing or non-absolute target: %s\n' "$target"
    exit 2
  fi
done
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$bundle_dir/../../.." && pwd)
: "${CUDA_HOME:?Set CUDA_HOME to a CUDA 12.8 toolkit with nvcc before installation}"
"$CUDA_HOME/bin/nvcc" --version
python3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
python3.12 -m venv "$venv_path"
python_bin="$venv_path/bin/python"
export PATH="$venv_path/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS="${MAX_JOBS:-4}"
"$python_bin" -m pip install -c "$bundle_dir/constraints.txt" pip setuptools wheel packaging ninja psutil
"$python_bin" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128
"$python_bin" -m pip install --no-build-isolation -c "$bundle_dir/constraints.txt" -r "$bundle_dir/requirements.txt"
git clone --filter=blob:none --no-checkout https://github.com/alibaba/ROLL.git "$adapter_path"
git -C "$adapter_path" checkout --detach 192b1a01ea61c113b2deb543f7b115783038dff8
"$python_bin" "$bundle_dir/prepare_adapter.py" "$adapter_path" --apply
"$python_bin" -m pip install -c "$bundle_dir/constraints.txt" -e "$adapter_path/mcore_adapter" -e "$repo_dir"
"$python_bin" -m pip check
printf 'Installation completed. Run preflight before training; see README_512k.md.\n'

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'Usage: bash run.sh /venv /patched-adapter /config.yaml\n'
  exit 2
fi
venv_path=$(realpath -e -- "$1")
adapter_path=$(realpath -e -- "$2")
config_path=$(realpath -e -- "$3")
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$bundle_dir/../../.." && pwd)
export PATH="$venv_path/bin:$PATH"
export PYTHONPATH="$repo_dir/src:$adapter_path/mcore_adapter/src${PYTHONPATH:+:$PYTHONPATH}"
library_paths=$("$venv_path/bin/python" -c 'import pathlib,site,sys; paths=[pathlib.Path(sys.prefix)/"lib"]; paths.extend(path for base in site.getsitepackages() for path in pathlib.Path(base).glob("nvidia/*/lib")); print(":".join(str(path) for path in paths if path.is_dir()))')
export LD_LIBRARY_PATH="$library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_MCA=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd -- "$repo_dir"
"$venv_path/bin/python" "$bundle_dir/preflight.py" --adapter "$adapter_path" --config "$config_path" --check-data
exec "$venv_path/bin/llamafactory-cli" train "$config_path"

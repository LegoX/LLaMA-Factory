#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'Usage: bash run.sh /env-prefix /config.yaml\n'
  printf 'Multi-node: export NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT before launching on every node.\n'
  exit 2
fi
env_path=$(realpath -e -- "$1")
config_path=$(realpath -e -- "$2")
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$bundle_dir/../../.." && pwd)
export PATH="$env_path/bin:$PATH"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
library_paths=$("$env_path/bin/python" -c 'import pathlib,site,sys; paths=[pathlib.Path(sys.prefix)/"lib"]; paths.extend(path for base in site.getsitepackages() for path in pathlib.Path(base).glob("nvidia/*/lib")); print(":".join(str(path) for path in paths if path.is_dir()))')
export LD_LIBRARY_PATH="$library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_MCA=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
cd -- "$repo_dir"
"$env_path/bin/python" "$bundle_dir/preflight.py" --config "$config_path" --check-data
exec "$env_path/bin/llamafactory-cli" train "$config_path"

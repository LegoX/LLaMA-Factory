#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf 'Usage: bash run.sh /env-prefix /config.yaml [key=value ...]\n'
  printf 'Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT on every node.\n'
  exit 2
fi
env_path=$(realpath -e -- "$1")
config_path=$(realpath -e -- "$2")
shift 2
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$bundle_dir/../../.." && pwd)
source "$bundle_dir/activate_shared.sh" "$env_path"
export NNODES="${NNODES:-1}"
if [[ $NNODES == 1 ]]; then
  export NODE_RANK=0 MASTER_ADDR=127.0.0.1
  unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE RDZV_ID MIN_NNODES MAX_NNODES
fi
cd -- "$repo_dir"
if [[ ${DRY_RUN:-0} == 1 ]]; then
  "$env_path/bin/python" "$bundle_dir/preflight.py" --config "$config_path" --config-only -- "$@"
  printf '[dry-run] '
  printf '%q ' "$env_path/bin/python" -m llamafactory.cli train "$config_path" "$@"
  printf '\n'
  exit 0
fi
"$env_path/bin/python" "$bundle_dir/preflight.py" --config "$config_path" --check-data -- "$@"
[[ ${PREFLIGHT_ONLY:-0} == 1 ]] && exit 0
if [[ ${MEMORY_MONITOR:-0} == 1 ]]; then
  "$env_path/bin/python" -u tools/monitor_training_memory.py --parent-pid "$$" \
    --interval "${MEMORY_MONITOR_INTERVAL:-15}" &
fi
exec "$env_path/bin/python" -m llamafactory.cli train "$config_path" "$@"

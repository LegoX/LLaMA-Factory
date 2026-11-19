#!/usr/bin/env bash
set -eo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $root/train.sh TASK.yaml [ACCELERATE.yaml] [--dry-run | --check] | --check-env"
  exit 2
fi
# Legacy model names select no separate interpreter; the YAML selects the model.
if [[ "$1" == qwen36 || "$1" == qwen38 ]]; then shift; fi
if [[ $# -lt 1 ]]; then echo "Missing YAML or --check-env" >&2; exit 2; fi
profile=qwen38
python_bin=/usr/local/share/venvs/qwen38/bin/python
cann_env="${CANN_ENV_FILE:-/usr/local/Ascend/cann-9.0.0/set_env.sh}"
if [[ ! -f "$cann_env" ]]; then echo "CANN environment missing: $cann_env" >&2; exit 2; fi
source "$cann_env"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="$(dirname -- "$python_bin"):$PATH"
if [[ "$profile" == qwen38 ]]; then export VIRTUAL_ENV=/usr/local/share/venvs/qwen38; fi
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export IMAGE_PLACEHOLDER='<|lf_media_image|>' VIDEO_PLACEHOLDER='<|lf_media_video|>' AUDIO_PLACEHOLDER='<|lf_media_audio|>'
unset USE_MCA USE_MEGATRON_BRIDGE
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
if [[ "$1" == --check-env ]]; then
  exec "$python_bin" "$root/bin/check_environment.py" "$profile"
fi
task_yaml="$(realpath -e -- "$1")"
shift
# Scratch is recreated on each node; it never contains the required model/code.
scratch="${NPU_TRAINING_SCRATCH:-/tmp/npu-training}"
mkdir -p "$scratch/runtime"
scratch="$(cd -- "$scratch" && pwd)"
export TRITON_CACHE_DIR="$scratch/triton" TORCH_EXTENSIONS_DIR="$scratch/extensions"
export HF_HOME="$scratch/huggingface" XDG_CACHE_HOME="$scratch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NPU_TRAINING_SCRATCH="$scratch"
exec "$python_bin" "$root/bin/launcher.py" "$task_yaml" "$@"

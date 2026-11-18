#!/usr/bin/env bash
set -eo pipefail
kit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]]; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
exec python "$kit_root/runner.py" "$@"

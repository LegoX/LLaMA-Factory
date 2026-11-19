#!/usr/bin/env bash
set -eo pipefail
exec bash /usr/local/share/npu-training/train.sh "$@"

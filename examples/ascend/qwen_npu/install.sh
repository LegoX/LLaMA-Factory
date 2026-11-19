#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 0 ]]; then echo 'Install location is fixed: /usr/local/share/npu-training' >&2; exit 2; fi
/usr/local/share/venvs/qwen38/bin/python - "$source_root" <<'PY'
from pathlib import Path
import shutil,sys
src=Path(sys.argv[1])/'image';dst=Path('/usr/local/share/npu-training')
for name in ['bin','configs']:shutil.copytree(src/name,dst/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(src/'train.sh',dst/'train.sh')
shutil.copy2(Path(sys.argv[1])/'README.md',dst/'README.md')
compat=Path('/usr/local/share/llamafactory-training');compat.mkdir(exist_ok=True)
shutil.copy2(Path(sys.argv[1])/'train.sh',compat/'train.sh')
print('Installed unified launcher:',dst)
PY

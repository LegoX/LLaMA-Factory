#!/usr/bin/env bash
set -euo pipefail
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target_root="${1:-/usr/local/share/llamafactory-training}"
python - "$source_root" "$target_root" <<'PY'
from pathlib import Path
import shutil,sys
src,dst=map(lambda s:Path(s).resolve(),sys.argv[1:])
if src==dst:raise SystemExit('Source and destination must differ')
dst.mkdir(parents=True,exist_ok=True)
for name in ['runtime','configs']:
 shutil.copytree(src/name,dst/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['train.sh','runner.py','LICENSE','README.md']:
 if (src/name).exists():shutil.copy2(src/name,dst/name)
print('Installed training code:',dst)
PY

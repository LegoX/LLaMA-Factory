# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrap src/train.py so trace_hook installs its monkey-patches at process start.

Set TRACE_STAGES=0 to bypass the patches entirely (equivalent to running
src/train.py directly).
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_TRAIN = _REPO_ROOT / "src" / "train.py"

if os.environ.get("TRACE_STAGES", "1") == "1":
    # ensure repo root on sys.path so `import tools.trace_hook` resolves
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import tools.trace_hook  # noqa: F401  side-effect: install patches

# emulate `python src/train.py <argv...>` so train.py sees __name__ == "__main__"
sys.argv[0] = str(_TRAIN)
runpy.run_path(str(_TRAIN), run_name="__main__")

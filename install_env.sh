#!/usr/bin/env bash
#
# install_env.sh — 一键安装 Qwen3.5-35B-A3B-Base SFT 训练环境
#
# 合并自:
#   - install.md                                  (基础安装步骤)
#   - docs/qwen3_5_moe_sft_troubleshooting.md      (7+1 个 bug 的修复 / "整体修复清单")
#
# 覆盖的 bug:
#   Bug 3  flash-linear-attention==0.5.0   (默认值,可通过 FLA_VERSION 覆盖)
#   Bug 4  transformers FA2 s_aux=None     (site-packages 一行补丁,本脚本自动打)
#   Bug 5  liger qwen3_5_moe dispatch      (已在仓库源码,pip install -e . 自动生效)
#   Bug 6  tilelang                        (pip,Hopper + Triton>=3.4 反向所需)
#   Bug 7  liger swiglu/rms_norm 关闭      (已在仓库源码)
#   Bug 8j torch==2.10.0                   (多机 hang 根治)
#
# 用法:
#   bash install_env.sh                 # 默认:创建/复用 conda env "lf_v3"
#   ENV_NAME=lf_v3 bash install_env.sh  # 自定义环境名
#
# 幂等:可重复运行;已装的包/已打的补丁会跳过。
set -euo pipefail

# ----------------------------------------------------------------------------
# 0. 配置
# ----------------------------------------------------------------------------
# 仓库根目录:默认取本脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-lf_v3}"
PY_VERSION="${PY_VERSION:-3.12}"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.6.0}"
FLA_VERSION="${FLA_VERSION:-0.5.0}"
FSSPEC_VERSION="${FSSPEC_VERSION:-2025.3.0}"

log()  { printf '\033[1;32m[install_env]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install_env]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[install_env] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# 1. 定位并初始化 conda
# ----------------------------------------------------------------------------
log "定位 conda ..."
# 优先用 PATH 里的 conda;否则可用 CONDA_HOME 指定安装目录,再退回 $HOME/miniconda3
if ! command -v conda >/dev/null 2>&1; then
    for guess in "${CONDA_HOME:-}" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [ -n "$guess" ] && [ -x "$guess/bin/conda" ]; then
            export PATH="$guess/bin:$PATH"
            break
        fi
    done
fi
command -v conda >/dev/null 2>&1 || die "找不到 conda。请先安装 Miniconda 并 conda init,或设置 CONDA_HOME。"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
log "conda base: $CONDA_BASE"

# ----------------------------------------------------------------------------
# 2. 创建 / 复用环境
# ----------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "环境 '$ENV_NAME' 已存在,复用。"
else
    log "创建环境 '$ENV_NAME' (python=$PY_VERSION) ..."
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

conda activate "$ENV_NAME"
PYBIN="$(command -v python)"
PIP="$PYBIN -m pip"
log "使用 python: $PYBIN"
[ -d "$REPO_DIR" ] || die "找不到仓库目录 $REPO_DIR"
cd "$REPO_DIR"

# ----------------------------------------------------------------------------
# 3. PyTorch 2.10.0 + cu128  (Bug 8j: 多机训练 hang 根治)
# ----------------------------------------------------------------------------
log "安装 PyTorch $TORCH_VERSION (cu128) ..."
$PIP install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$TORCH_INDEX_URL"

# ----------------------------------------------------------------------------
# 4. LlamaFactory 本体 + 依赖
#    (Bug 5 / Bug 7 的 liger dispatch 补丁已在仓库源码里,-e 安装自动生效)
# ----------------------------------------------------------------------------
log "安装 LlamaFactory (editable) ..."
$PIP install --index-url "$PYPI_INDEX_URL" -e .

log "安装 requirements (metrics / deepspeed / liger-kernel) ..."
$PIP install --index-url "$PYPI_INDEX_URL" -r requirements/metrics.txt
$PIP install --index-url "$PYPI_INDEX_URL" -r requirements/deepspeed.txt
$PIP install --index-url "$PYPI_INDEX_URL" -r requirements/liger-kernel.txt

# ----------------------------------------------------------------------------
# 5. flash-attn (FA2)
# ----------------------------------------------------------------------------
log "安装 flash-attn (--no-build-isolation,编译较慢) ..."
$PIP install \
    --index-url "$PYPI_INDEX_URL" \
    flash-attn \
    --no-build-isolation

# ----------------------------------------------------------------------------
# 6. Qwen3.5 linear-attention 依赖与版本纠偏
#    直接依赖已由 LlamaFactory 安装。这里使用 --no-deps,避免把 CUDA 12.8
#    的 PyTorch 替换成 PyPI 上其它 CUDA 版本,也避免 transformers 漂移。
# ----------------------------------------------------------------------------
log "固定 transformers / flash-linear-attention / fsspec 版本 ..."
$PIP install \
    --index-url "$PYPI_INDEX_URL" \
    --no-deps \
    --upgrade \
    "flash-linear-attention==$FLA_VERSION" \
    "fla-core==$FLA_VERSION" \
    "transformers==$TRANSFORMERS_VERSION" \
    "fsspec==$FSSPEC_VERSION"

# ----------------------------------------------------------------------------
# 7. Bug 6: tilelang (Hopper + Triton>=3.4 上 gated_delta_rule 反向必需)
# ----------------------------------------------------------------------------
log "安装 tilelang (Bug 6) ..."
$PIP install --index-url "$PYPI_INDEX_URL" tilelang

# ----------------------------------------------------------------------------
# 8. 实验追踪
# ----------------------------------------------------------------------------
log "安装 wandb ..."
$PIP install --index-url "$PYPI_INDEX_URL" wandb

# ----------------------------------------------------------------------------
# 9. Bug 4: transformers FA2 s_aux=None 解引用补丁
#    transformers <5.7 在 flash_attention.py 无条件 s_aux.to(...);视觉塔前向
#    时 s_aux=None -> 'NoneType' has no attribute 'to'。这是 site-packages 里的
#    包,重装 transformers 会丢补丁,故每次安装后都自动(幂等)重打。
# ----------------------------------------------------------------------------
log "应用 Bug 4 补丁:transformers FA2 s_aux=None guard ..."
"$PYBIN" - <<'PYEOF'
import os
import transformers

fa = os.path.join(os.path.dirname(transformers.__file__),
                  "integrations", "flash_attention.py")
if not os.path.exists(fa):
    print(f"[install_env]   跳过:找不到 {fa}")
    raise SystemExit(0)

src = open(fa, encoding="utf-8").read()
buggy = "s_aux=s_aux.to(query.dtype),"
fixed = "s_aux=s_aux.to(query.dtype) if s_aux is not None else None,"

if fixed in src:
    print(f"[install_env]   已是修复版({transformers.__version__}),无需改动。")
elif buggy in src:
    open(fa, "w", encoding="utf-8").write(src.replace(buggy, fixed, 1))
    print(f"[install_env]   已打补丁 -> {fa}")
else:
    # 上游 >=5.7 已修复,或写法不同;不强改,提示人工确认。
    print(f"[install_env]   未找到目标行,可能上游已修复({transformers.__version__})。"
          " 若训练时报 's_aux NoneType.to'，请手动核对 flash_attention.py。")
PYEOF

# ----------------------------------------------------------------------------
# 10. 校验
# ----------------------------------------------------------------------------
log "检查 Python 包依赖一致性 ..."
$PIP check

log "环境自检 ..."
EXPECTED_TORCH_VERSION="$TORCH_VERSION" \
EXPECTED_TRANSFORMERS_VERSION="$TRANSFORMERS_VERSION" \
EXPECTED_FLA_VERSION="$FLA_VERSION" \
EXPECTED_FSSPEC_VERSION="$FSSPEC_VERSION" \
"$PYBIN" - <<'PYEOF'
import importlib
import os

import torch


def version(module):
    imported = importlib.import_module(module)
    return getattr(imported, "__version__", "ok")


modules = [
    ("transformers", "transformers"),
    ("llamafactory", "llamafactory"),
    ("flash_attn", "flash_attn"),
    ("fla (flash-linear-attention)", "fla"),
    ("fsspec", "fsspec"),
    ("liger_kernel", "liger_kernel"),
    ("tilelang", "tilelang"),
    ("deepspeed", "deepspeed"),
    ("wandb", "wandb"),
]
versions = {"torch": torch.__version__}

print(f"  torch                  : {torch.__version__}")
print(f"  torch.cuda.is_available: {torch.cuda.is_available()}")
try:
    print(f"  bundled NCCL           : {torch.cuda.nccl.version()}")
except Exception:
    pass
for label, module in modules:
    versions[module] = version(module)
    print(f"  {label:<23}: {versions[module]}")

expected = {
    "torch": os.environ["EXPECTED_TORCH_VERSION"],
    "transformers": os.environ["EXPECTED_TRANSFORMERS_VERSION"],
    "fla": os.environ["EXPECTED_FLA_VERSION"],
    "fsspec": os.environ["EXPECTED_FSSPEC_VERSION"],
}
actual_torch = versions["torch"].split("+", 1)[0]
if actual_torch != expected["torch"]:
    raise RuntimeError(f"torch version drift: expected {expected['torch']}, got {versions['torch']}")
for module in ("transformers", "fla", "fsspec"):
    if versions[module] != expected[module]:
        raise RuntimeError(
            f"{module} version drift: expected {expected[module]}, got {versions[module]}"
        )

# 导入执行 LLaMA-Factory 依赖版本检查的解析器路径。
importlib.import_module("llamafactory.hparams")
print("  LLaMA-Factory dependency check: OK")
PYEOF

log "完成 ✅  环境 '$ENV_NAME' 已就绪。"
cat <<EOF

下一步:
  conda activate $ENV_NAME
  cd $REPO_DIR

  # 单机 8 卡
  bash run_sft_qwen3_5_35b_a3b_base.sh

  # 多机 (以 4 节点为例,每台机器各跑一次,NODE_RANK 取 0..3)
  NNODES=4 NODE_RANK=0 bash run_sft_qwen3_5_35b_a3b_base.sh
EOF

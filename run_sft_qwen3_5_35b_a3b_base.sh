#!/usr/bin/env bash
#
# 用法示例:
#
# 1) 单节点 8 卡:
#   cd /jyx_data/LLaMA-Factory-latest
#   bash run_sft_qwen3_5_35b_a3b_base.sh
#
# 2) 两节点各 8 卡(推荐显式指定 node0 可被 node1 访问的 IP):
#   # node0
#   cd /jyx_data/LLaMA-Factory-latest
#   NNODES=2 NODE_RANK=0 bash run_sft_qwen3_5_35b_a3b_base.sh
#
#   # node1
#   cd /jyx_data/LLaMA-Factory-latest
#   NNODES=2 NODE_RANK=1 bash run_sft_qwen3_5_35b_a3b_base.sh
#
# 3) 四节点各 8 卡:
#   # node0
#   cd /jyx_data/LLaMA-Factory-latest
#   NNODES=4 NODE_RANK=0 bash run_sft_qwen3_5_35b_a3b_base.sh
#
#   # node1
#   cd /jyx_data/LLaMA-Factory-latest
#   NNODES=4 NODE_RANK=1 bash run_sft_qwen3_5_35b_a3b_base.sh
#
#   # node2
#   cd /jyx_data/LLaMA-Factory-latest
#   NNODES=4 NODE_RANK=2 bash run_sft_qwen3_5_35b_a3b_base.sh
#
#   # node3
#   cd /jyx_data/LLaMA-Factory-latest
#   NNODES=4 NODE_RANK=3 bash run_sft_qwen3_5_35b_a3b_base.sh
#
#   # 若 /jyx_data/LLaMA-Factory-latest 不是四节点共享目录,建议四台机器显式传同一个 MASTER_ADDR=<node0_ip>。
#
# 说明:
# - 启动前需 export WANDB_API_KEY=xxx (脚本不再内置默认值)。
# - 默认 W&B run_name 取 TRAIN_CONFIG 文件名(去掉后缀)并追加时间戳,可通过 RUN_NAME=xxx 覆盖。
# - 默认使用 NPROC_PER_NODE=8, TARGET_GBS=64, PER_DEVICE_BS=1。
# - 默认在 torchrun 前清理所有用户的 GPU compute 进程; 需要足够权限。
# - 如需跳过清理: CLEAR_GPU_PROCS=0 bash run_sft_qwen3_5_35b_a3b_base.sh
# - 多网卡环境建议额外指定 NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME。

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# 经过 8a/8b/8c/8d 多轮验证,本机房 RoCE fabric 上瞬态停顿是分钟级,
# IB QP 重试预算(IB_TIMEOUT × IB_RETRY_CNT ≈ 245s)扛不过,QP 死掉后 NCCL 永远等不到 ACK;
# 反而纯 socket-on-eth0 由 kernel TCP 自己擦屁股(8b 撑了 6h44m / 388 步 vs 8c/8d 仅 ~12 min)。
# ZeRO-3 + gradient checkpointing 的算/通比下,RoCE 那点带宽优势在 backward 里被算力盖掉,
# 实测每步 62s 跟 socket 一致——所以 A 方案: 关 IB,走 socket,稳定性优先。
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set; e.g. export WANDB_API_KEY=...}

# === 诊断仪表(供应商要求 5/18: 抓全 traceback,定位前向/反向/数据切片/保存阶段) ===
# Python 不缓冲 + faulthandler 自动 dump py 栈(SIGABRT/SIGSEGV/SIGFPE 都触发)
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
# C++ 栈
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-INFO}
# NCCL hang 时定位具体 collective:本轮一次性提到 INFO + 关键子系统,稳了再降回 WARN
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL,NET}
# 阶段级插桩(tools/trace_hook.py): dataloader/forward/backward/save 各打 enter/exit
export TRACE_STAGES=${TRACE_STAGES:-1}
# collective 级插桩刷屏,默认关;hang 现场再开
export TRACE_COLLECTIVES=${TRACE_COLLECTIVES:-0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-info}

# OOB + 数据面统一走 eth0。多网卡环境若 NCCL 自动探测错网卡(走到 docker0/veth1/lo)
# 极易导致跨机集合通信 hang/超时(4 节点 run 20260516_004703 在 step 69 即此症状)。
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}

# 明确禁掉 IB 探测,不让 NCCL 再尝试走 mlx5_bond_*。配合上面的 SOCKET_IFNAME 就是 socket 通路。
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# socket 模式下并行收发线程,压尾延迟。8 × 8 = 64 路 socket 已足够喂满 100/200G eth0。
# export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-8}
# export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-8}

# watchdog/诊断: TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 控制 monitor thread 在 watchdog
# 触发后等多久才 SIGABRT 进程(从而让 torchrun 退出、wrapper 进入 retry)。
# 注意: 不是"trace dump 容忍度"——而是"hang 之后再卡多少秒才放手"。
# 8h 实测: 之前设 1800s 让 20260517_093118 在 fire 后又空转半小时 retry 才起。
# 健康 rank 几秒就能 dump 完;卡死 rank 等再久也不会 dump。180s 足够留 margin。
# 8d 一度 patch 过 deepspeed/comm/torch.py 给子 PG 改 1800s,8f run(20260516_194508)证实
# collective 跑满 1800s 也没自愈,那次 patch 纯粹白等 1200s——已撤回(参见 docs Bug 8g)。
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-180}
export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE:-2097152}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}

# Pure-text SFT on a multimodal-capable base model (Qwen3.5-MoE):
# the default <image>/<video>/<audio> placeholders collide with literal substrings
# that appear in code/issue text (JSX, HTML, markdown). Override to sentinels that
# will not occur in the data so mm_plugin._validate_messages always sees count=0.
export IMAGE_PLACEHOLDER=${IMAGE_PLACEHOLDER:-"<|__lf_image_placeholder__|>"}
export VIDEO_PLACEHOLDER=${VIDEO_PLACEHOLDER:-"<|__lf_video_placeholder__|>"}
export AUDIO_PLACEHOLDER=${AUDIO_PLACEHOLDER:-"<|__lf_audio_placeholder__|>"}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

query_gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF {print $1}' | sort -u
}

pid_in_list() {
    local needle=$1
    shift
    local pid
    for pid in "$@"; do
        if [ "$pid" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

clear_gpu_procs() {
    if [ "${CLEAR_GPU_PROCS:-1}" != "1" ]; then
        return 0
    fi

    GPU_CLEANUP_TERM_WAIT=${GPU_CLEANUP_TERM_WAIT:-10}
    if ! is_uint "$GPU_CLEANUP_TERM_WAIT"; then
        echo "[error] GPU_CLEANUP_TERM_WAIT must be a non-negative integer"
        exit 1
    fi

    echo "[gpu-cleanup] CLEAR_GPU_PROCS=1, CLEAR_GPU_PROCS_ALL_USERS=${CLEAR_GPU_PROCS_ALL_USERS:-1}, checking existing GPU compute processes..."
    mapfile -t gpu_pids < <(query_gpu_pids)

    if [ "${#gpu_pids[@]}" -eq 0 ]; then
        echo "[gpu-cleanup] no GPU compute processes found"
        return 0
    fi

    current_uid=$(id -u)
    killed_pids=()
    for pid in "${gpu_pids[@]}"; do
        [ -n "$pid" ] || continue
        kill -0 "$pid" 2>/dev/null || continue

        owner_uid=$(stat -c %u "/proc/$pid" 2>/dev/null || true)
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
        if [ "${CLEAR_GPU_PROCS_ALL_USERS:-1}" != "1" ] && [ "$owner_uid" != "$current_uid" ]; then
            echo "[gpu-cleanup] skip pid=$pid owner_uid=$owner_uid; set CLEAR_GPU_PROCS_ALL_USERS=1 to include it"
            continue
        fi

        echo "[gpu-cleanup] terminate pid=$pid owner_uid=$owner_uid cmd=${cmd:0:160}"
        kill "$pid" 2>/dev/null || echo "[gpu-cleanup] warn: failed to terminate pid=$pid, will try SIGKILL later"
        killed_pids+=("$pid")
    done

    if [ "${#killed_pids[@]}" -eq 0 ]; then
        echo "[gpu-cleanup] no eligible GPU processes were terminated"
        return 0
    fi

    waited=0
    while [ "$waited" -lt "$GPU_CLEANUP_TERM_WAIT" ]; do
        mapfile -t remaining_gpu_pids < <(query_gpu_pids)
        alive=0
        for pid in "${killed_pids[@]}"; do
            if pid_in_list "$pid" "${remaining_gpu_pids[@]}"; then
                alive=1
                break
            fi
        done
        [ "$alive" -eq 0 ] && break
        sleep 1
        waited=$((waited + 1))
    done

    mapfile -t remaining_gpu_pids < <(query_gpu_pids)
    for pid in "${killed_pids[@]}"; do
        if pid_in_list "$pid" "${remaining_gpu_pids[@]}"; then
            echo "[gpu-cleanup] force kill pid=$pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    sleep 2
    mapfile -t remaining_gpu_pids < <(query_gpu_pids)
    cleanup_failed=0
    for pid in "${killed_pids[@]}"; do
        if pid_in_list "$pid" "${remaining_gpu_pids[@]}"; then
            echo "[error] GPU process pid=$pid is still reported by nvidia-smi after cleanup; run with sufficient permissions or reset/reboot the node"
            cleanup_failed=1
        fi
    done

    if [ "$cleanup_failed" -ne 0 ]; then
        exit 1
    fi
}

# 分布式参数(可通过 env 覆盖)
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}
TRAIN_CONFIG=${TRAIN_CONFIG:-examples/train_full/qwen3_5_35b_a3b_base_pangu_code_data_0417_gbs64pbs1acc8_lr5e-5_epo3.yaml}
TRAIN_CONFIG_BASENAME=$(basename "$TRAIN_CONFIG")
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN_NAME=${RUN_NAME:-${TRAIN_CONFIG_BASENAME%.*}_${RUN_TIMESTAMP}}

# NCCL flight recorder dump 落到本仓库 logs/(watchdog 触发时每 rank 一个 pickle)
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-$SCRIPT_DIR/logs/nccl_trace_${RUN_TIMESTAMP}_node${NODE_RANK}_rank}"
mkdir -p "$SCRIPT_DIR/logs"

for name in NNODES NODE_RANK NPROC_PER_NODE MASTER_PORT; do
    if ! is_uint "${!name}"; then
        echo "[error] $name must be a non-negative integer"
        exit 1
    fi
done

if [ "$NNODES" -lt 1 ] || [ "$NPROC_PER_NODE" -lt 1 ]; then
    echo "[error] NNODES and NPROC_PER_NODE must be >= 1"
    exit 1
fi

if [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "[error] NODE_RANK=$NODE_RANK is out of range for NNODES=$NNODES"
    exit 1
fi

if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "[error] training config not found: $TRAIN_CONFIG"
    exit 1
fi

# 自动续训:扫描 YAML 的 output_dir,挑步数最大且含 trainer_state.json 的 checkpoint-* 目录。
# - RESUME=0           关闭自动续训
# - RESUME_FROM_CHECKPOINT=<path>  显式指定路径(跳过自动检测)
RESUME=${RESUME:-1}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

if [ "$RESUME" = "1" ] && [ -z "$RESUME_FROM_CHECKPOINT" ]; then
    OUTPUT_DIR=$(awk '
        /^[[:space:]]*output_dir[[:space:]]*:/ {
            sub(/^[^:]*:[[:space:]]*/, "")
            sub(/[[:space:]]*#.*$/, "")
            gsub(/["'\'']/, "")
            sub(/[[:space:]]+$/, "")
            print
            exit
        }' "$TRAIN_CONFIG")

    if [ -z "$OUTPUT_DIR" ]; then
        echo "[resume] could not parse output_dir from $TRAIN_CONFIG, skip auto-detect"
    elif [ ! -d "$OUTPUT_DIR" ]; then
        echo "[resume] output_dir '$OUTPUT_DIR' does not exist, will train from scratch"
    else
        latest_ckpt=""
        latest_step=-1
        for d in "$OUTPUT_DIR"/checkpoint-*; do
            [ -d "$d" ] || continue
            [ -f "$d/trainer_state.json" ] || continue
            step=${d##*/checkpoint-}
            is_uint "$step" || continue
            if [ "$step" -gt "$latest_step" ]; then
                latest_step=$step
                latest_ckpt=$d
            fi
        done
        if [ -n "$latest_ckpt" ]; then
            RESUME_FROM_CHECKPOINT=$latest_ckpt
            echo "[resume] auto-detected latest checkpoint: $RESUME_FROM_CHECKPOINT (step=$latest_step)"
        else
            echo "[resume] no valid checkpoint under $OUTPUT_DIR, will train from scratch"
        fi
    fi
fi

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
        echo "[error] RESUME_FROM_CHECKPOINT='$RESUME_FROM_CHECKPOINT' is not a directory"
        exit 1
    fi
    EXTRA_ARGS+=("resume_from_checkpoint=$RESUME_FROM_CHECKPOINT")
fi

# MASTER_ADDR 自动协商:多节点时,node0 把自己 IP 写到共享文件,其他节点读
# 显式 export MASTER_ADDR 可跳过此逻辑
RDZV_FILE=${RDZV_FILE:-$SCRIPT_DIR/.rdzv_master_addr_${MASTER_PORT}}
RDZV_TIMEOUT=${RDZV_TIMEOUT:-300}
RDZV_MAX_AGE=${RDZV_MAX_AGE:-600}
for name in RDZV_TIMEOUT RDZV_MAX_AGE; do
    if ! is_uint "${!name}" || [ "${!name}" -lt 1 ]; then
        echo "[error] $name must be a positive integer"
        exit 1
    fi
done

if [ -z "${MASTER_ADDR:-}" ]; then
    if [ "$NNODES" -le 1 ]; then
        MASTER_ADDR=127.0.0.1
    elif [ "$NODE_RANK" = "0" ]; then
        rm -f "$RDZV_FILE"
        MASTER_ADDR=$(hostname -I | awk '{print $1}')
        if [ -z "$MASTER_ADDR" ]; then
            echo "[error] failed to infer MASTER_ADDR; please export MASTER_ADDR explicitly"
            exit 1
        fi
        echo "$MASTER_ADDR" > "$RDZV_FILE"
        echo "[rdzv] node0 wrote MASTER_ADDR=$MASTER_ADDR to $RDZV_FILE"
    else
        echo "[rdzv] node$NODE_RANK waiting for $RDZV_FILE (timeout=${RDZV_TIMEOUT}s)..."
        for i in $(seq 1 "$RDZV_TIMEOUT"); do
            if [ -s "$RDZV_FILE" ]; then
                file_age=$(( $(date +%s) - $(stat -c %Y "$RDZV_FILE") ))
                [ "$file_age" -le "$RDZV_MAX_AGE" ] && break
            fi
            sleep 1
        done
        if [ ! -s "$RDZV_FILE" ]; then
            echo "[error] $RDZV_FILE not available after ${RDZV_TIMEOUT}s, abort"
            exit 1
        fi
        file_age=$(( $(date +%s) - $(stat -c %Y "$RDZV_FILE") ))
        if [ "$file_age" -gt "$RDZV_MAX_AGE" ]; then
            echo "[error] $RDZV_FILE is stale (${file_age}s old), abort"
            exit 1
        fi
        read -r MASTER_ADDR < "$RDZV_FILE"
        echo "[rdzv] node$NODE_RANK read MASTER_ADDR=$MASTER_ADDR from $RDZV_FILE"
    fi
fi

# 根据总卡数自动调整 gradient_accumulation_steps,保持目标 global batch size 不变
# 公式: grad_accum = TARGET_GBS / (PER_DEVICE_BS * NPROC_PER_NODE * NNODES)
TARGET_GBS=${TARGET_GBS:-64}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
for name in TARGET_GBS PER_DEVICE_BS; do
    if ! is_uint "${!name}"; then
        echo "[error] $name must be a non-negative integer"
        exit 1
    fi
done
if [ "$PER_DEVICE_BS" -lt 1 ] || [ "$TARGET_GBS" -lt 1 ]; then
    echo "[error] TARGET_GBS and PER_DEVICE_BS must be >= 1"
    exit 1
fi
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
DENOM=$((PER_DEVICE_BS * TOTAL_GPUS))
if [ $((TARGET_GBS % DENOM)) -ne 0 ]; then
    echo "[error] TARGET_GBS=$TARGET_GBS 不能被 PER_DEVICE_BS*TOTAL_GPUS=$DENOM 整除,请调整 TARGET_GBS / PER_DEVICE_BS"
    exit 1
fi
GRAD_ACCUM=$((TARGET_GBS / DENOM))

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"

echo "[dist] NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[bs]   TARGET_GBS=$TARGET_GBS PER_DEVICE_BS=$PER_DEVICE_BS TOTAL_GPUS=$TOTAL_GPUS -> gradient_accumulation_steps=$GRAD_ACCUM"
echo "[cfg]  TRAIN_CONFIG=$TRAIN_CONFIG"
echo "[wandb] RUN_NAME=$RUN_NAME"
echo "[log]  TRAIN_LOG=$TRAIN_LOG"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "[extra] ${EXTRA_ARGS[*]}"
fi

clear_gpu_procs
nvidia-smi

# 训练重试循环: 本机房 fabric MTBF ~4-7h(见 docs Bug 8e/8f),watchdog fire 后整个
# torchrun 退出。每次循环重新进入脚本顶部 RESUME 逻辑会从最近 checkpoint 自动续训。
# AUTO_RETRY=0 关闭自动重试; MAX_RETRIES 上限避免死循环(checkpoint 没动则停)。
AUTO_RETRY=${AUTO_RETRY:-1}
MAX_RETRIES=${MAX_RETRIES:-1}
RETRY_BACKOFF=${RETRY_BACKOFF:-30}

run_torchrun() {
    local tr_log_dir="$LOG_DIR/torchrun_${RUN_TIMESTAMP}_node${NODE_RANK}_attempt${attempt}"
    mkdir -p "$tr_log_dir"
    # TRACE_STAGES=1: tools/trace_train.py 先 import trace_hook 再 runpy src/train.py
    # TRACE_STAGES=0: 退化为原行为,直接跑 src/train.py
    local entry="src/train.py"
    if [ "${TRACE_STAGES:-1}" = "1" ] && [ -f "tools/trace_train.py" ]; then
        entry="tools/trace_train.py"
    fi

    # --tee 3 / --redirects 3: 全部 rank 的 stdout+stderr 既前缀回流到 console,
    # 又落盘到 $tr_log_dir/attempt_*/<local_rank>/{stdout,stderr}.log,方便事后 grep 单 rank
    # awk 给每行加毫秒时间戳;PIPESTATUS[0] 仍是 torchrun 真实 rc
    torchrun --nnodes="$NNODES" \
         --nproc_per_node="$NPROC_PER_NODE" \
         --master_addr="$MASTER_ADDR" \
         --master_port="$MASTER_PORT" \
         --node_rank="$NODE_RANK" \
         --tee 3 \
         --redirects 3 \
         --log-dir "$tr_log_dir" \
         "$entry" "$TRAIN_CONFIG" \
         run_name="$RUN_NAME" \
         per_device_train_batch_size="$PER_DEVICE_BS" \
         gradient_accumulation_steps="$GRAD_ACCUM" \
         "${EXTRA_ARGS[@]}" 2>&1 \
      | awk -W interactive '{ printf "%s %s\n", strftime("%Y-%m-%dT%H:%M:%S", systime()), $0; fflush() }' \
      | tee -a "$TRAIN_LOG"
    return "${PIPESTATUS[0]}"
}

# 重新扫描最近 checkpoint(每次重试前); 跟脚本顶部同样逻辑,但单独抽出来调用
rescan_resume() {
    [ "$RESUME" = "1" ] || return 0
    local out_dir latest_ckpt latest_step step d
    out_dir=$(awk '
        /^[[:space:]]*output_dir[[:space:]]*:/ {
            sub(/^[^:]*:[[:space:]]*/, "")
            sub(/[[:space:]]*#.*$/, "")
            gsub(/["'\'']/, "")
            sub(/[[:space:]]+$/, "")
            print
            exit
        }' "$TRAIN_CONFIG")
    [ -d "$out_dir" ] || return 0
    latest_ckpt=""
    latest_step=-1
    for d in "$out_dir"/checkpoint-*; do
        [ -d "$d" ] || continue
        [ -f "$d/trainer_state.json" ] || continue
        step=${d##*/checkpoint-}
        is_uint "$step" || continue
        if [ "$step" -gt "$latest_step" ]; then
            latest_step=$step
            latest_ckpt=$d
        fi
    done
    if [ -n "$latest_ckpt" ]; then
        # 替换 EXTRA_ARGS 里旧的 resume_from_checkpoint=...,没有就追加
        local found=0 i
        for i in "${!EXTRA_ARGS[@]}"; do
            case "${EXTRA_ARGS[$i]}" in
                resume_from_checkpoint=*)
                    EXTRA_ARGS[$i]="resume_from_checkpoint=$latest_ckpt"; found=1 ;;
            esac
        done
        [ $found -eq 0 ] && EXTRA_ARGS+=("resume_from_checkpoint=$latest_ckpt")
        echo "[retry] resume from $latest_ckpt (step=$latest_step)"
    fi
}

attempt=0
while : ; do
    attempt=$((attempt + 1))
    echo "[run] attempt=$attempt / max=$MAX_RETRIES"
    set +e
    run_torchrun
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[run] training finished cleanly (attempt=$attempt)"
        exit 0
    fi
    echo "[run] torchrun exited rc=$rc (attempt=$attempt)"
    if [ "$AUTO_RETRY" != "1" ]; then
        echo "[run] AUTO_RETRY=0, abort"
        exit "$rc"
    fi
    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
        echo "[run] hit MAX_RETRIES=$MAX_RETRIES, abort"
        exit "$rc"
    fi
    # 进入下次循环前: 清理 zombie GPU 进程 + 等 backoff + 重扫 ckpt
    clear_gpu_procs
    sleep "$RETRY_BACKOFF"
    rescan_resume
    TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"
    echo "[retry] new log: $TRAIN_LOG"
done

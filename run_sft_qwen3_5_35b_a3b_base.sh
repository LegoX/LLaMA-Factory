#!/usr/bin/env bash
#
# Full-parameter SFT launcher for Qwen3.5-35B-A3B-Base.
#
# Usage
#
# 1) Single node, 8 GPUs:
#   cd /path/to/LLaMA-Factory
#   bash run_sft_qwen3_5_35b_a3b_base.sh
#
# 2) Two nodes, 8 GPUs each:
#   # node0
#   NNODES=2 NODE_RANK=0 bash run_sft_qwen3_5_35b_a3b_base.sh
#   # node1
#   NNODES=2 NODE_RANK=1 bash run_sft_qwen3_5_35b_a3b_base.sh
#
# 3) Four nodes, 8 GPUs each: run once per node with NODE_RANK=0..3 and NNODES=4.
#
#   If the repository directory is not shared across nodes, the rendezvous file
#   cannot be used; export the same MASTER_ADDR=<node0_ip> on every node instead.
#
# Notes
# - Export WANDB_API_KEY before starting; the script has no built-in default and
#   refuses to run without it. Never hardcode credentials in this file.
# - The W&B run_name defaults to the TRAIN_CONFIG basename plus a timestamp;
#   override with RUN_NAME=xxx.
# - Defaults: NPROC_PER_NODE=8, TARGET_GBS=64, PER_DEVICE_BS=1.
# - By default all users' GPU compute processes are cleared before torchrun,
#   which requires sufficient permissions. Skip it with CLEAR_GPU_PROCS=0.
# - On hosts with several NICs, set NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME
#   explicitly to match your fabric.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[error] WANDB_API_KEY is required; export it before starting training." >&2
    exit 1
fi
export WANDB_API_KEY

# === Diagnostics ===
# Verbose tracing is off by default. DEBUG_HANG=1 switches the whole script into
# investigation mode (NCCL INFO, C++ log level INFO, per-rank torchrun logs) for
# diagnosing multi-node collective hangs.
DEBUG_HANG=${DEBUG_HANG:-0}

if [ "$DEBUG_HANG" = "1" ]; then
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL,NET}
    export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-INFO}
    export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-info}
else
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-}
    export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-WARNING}
    export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-warning}
fi

# Post-mortem forensics, kept in every mode: these cost nothing while the process
# is healthy and only produce output once it crashes.
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
# faulthandler: dump the Python stack on SIGABRT/SIGSEGV/SIGFPE
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
# print C++ stacks on crash (silent otherwise)
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}

# NCCL transport: socket on eth0 (IB disabled) + multi-socket + 8 MiB ring buffer.
# See docs/qwen3_5_moe_sft_multinode_notes.md for why these values were chosen.
# Override any of them to match your own fabric.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-8}
export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-8}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-8388608}

# Watchdog: once it fires, SIGABRT after 180s so torchrun exits into the auto-retry
# loop. Note this is not a "hang tolerance" — it is how long to keep waiting after a
# hang is already detected, so raising it only delays recovery.
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

# Distributed parameters (override via env)
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}
TRAIN_CONFIG=${TRAIN_CONFIG:-examples/train_full/qwen3_5_35b_a3b_base_selfmade_traj_selected_gbs64pbs1acc8_lr5e-5_epo3.yaml}
TRAIN_CONFIG_BASENAME=$(basename "$TRAIN_CONFIG")
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN_NAME=${RUN_NAME:-${TRAIN_CONFIG_BASENAME%.*}_${RUN_TIMESTAMP}}

# NCCL flight recorder dumps land in logs/ (one pickle per rank when the watchdog fires)
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

# Auto-resume: scan output_dir from the YAML and pick the highest-step checkpoint-*
# directory that contains a trainer_state.json.
# - RESUME=0                       disable auto-resume
# - RESUME_FROM_CHECKPOINT=<path>  use an explicit path (skips auto-detection)
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

# MASTER_ADDR rendezvous: on multi-node runs node0 writes its IP to a shared file
# and the other nodes read it. Export MASTER_ADDR explicitly to skip this.
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

# Derive gradient_accumulation_steps from the total GPU count so the global batch
# size stays fixed: grad_accum = TARGET_GBS / (PER_DEVICE_BS * NPROC_PER_NODE * NNODES)
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
    echo "[error] TARGET_GBS=$TARGET_GBS is not divisible by PER_DEVICE_BS*TOTAL_GPUS=$DENOM; adjust TARGET_GBS or PER_DEVICE_BS"
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

# Retry loop, a safety net for occasional collective hangs or flaky nodes: after the
# watchdog fires torchrun exits and the next iteration resumes from the latest
# checkpoint. AUTO_RETRY=0 disables it; MAX_RETRIES bounds the loop.
AUTO_RETRY=${AUTO_RETRY:-1}
MAX_RETRIES=${MAX_RETRIES:-20}
RETRY_BACKOFF=${RETRY_BACKOFF:-30}

run_torchrun() {
    local entry="src/train.py"

    # Normal mode (DEBUG_HANG=0): a single merged $TRAIN_LOG, every rank timestamped by awk.
    # Investigation mode (DEBUG_HANG=1): additionally enable torchrun --tee/--redirects so
    # each rank's stdout+stderr also lands in $tr_log_dir/<local_rank>/, making it possible
    # to grep a single rank after the fact.
    local tee_args=()
    if [ "${DEBUG_HANG:-0}" = "1" ]; then
        local tr_log_dir="$LOG_DIR/torchrun_${RUN_TIMESTAMP}_node${NODE_RANK}_attempt${attempt}"
        mkdir -p "$tr_log_dir"
        tee_args=(--tee 3 --redirects 3 --log-dir "$tr_log_dir")
    fi

    # PIPESTATUS[0] is still torchrun's real exit code
    torchrun --nnodes="$NNODES" \
         --nproc_per_node="$NPROC_PER_NODE" \
         --master_addr="$MASTER_ADDR" \
         --master_port="$MASTER_PORT" \
         --node_rank="$NODE_RANK" \
         "${tee_args[@]}" \
         "$entry" "$TRAIN_CONFIG" \
         run_name="$RUN_NAME" \
         per_device_train_batch_size="$PER_DEVICE_BS" \
         gradient_accumulation_steps="$GRAD_ACCUM" \
         "${EXTRA_ARGS[@]}" 2>&1 \
      | awk -W interactive '{ printf "%s %s\n", strftime("%Y-%m-%dT%H:%M:%S", systime()), $0; fflush() }' \
      | tee -a "$TRAIN_LOG"
    return "${PIPESTATUS[0]}"
}

# Rescan for the latest checkpoint before each retry (same logic as above, factored out)
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
        # replace any existing resume_from_checkpoint= in EXTRA_ARGS, otherwise append
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
    # before the next attempt: clear zombie GPU processes, back off, rescan checkpoints
    clear_gpu_procs
    sleep "$RETRY_BACKOFF"
    rescan_resume
    TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"
    echo "[retry] new log: $TRAIN_LOG"
done

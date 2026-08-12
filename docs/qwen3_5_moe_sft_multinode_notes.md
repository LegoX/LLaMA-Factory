# Qwen3.5-MoE SFT: multi-node and long-context notes

Reference notes for running full-parameter SFT of **Qwen3.5-35B-A3B-Base** at
`cutoff_len=131072` on H800-class hardware with DeepSpeed ZeRO-3 and FlashAttention 2.

Every dependency pin in [`install_env.sh`](../install_env.sh) and every environment
variable in [`run_sft_qwen3_5_35b_a3b_base.sh`](../run_sft_qwen3_5_35b_a3b_base.sh) exists
because of one of the failures below. This document explains which, so that the settings
can be adjusted deliberately rather than removed by guesswork.

For a step-by-step walkthrough of a normal run, see the
[SFT quickstart](qwen3_5_35b_sft_quickstart.md).

## Validated baseline

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.4.0 |
| CUDA | 12.8 |
| transformers | 5.6.0 |
| flash-linear-attention (fla) | 0.5.0 |
| liger-kernel | 0.8.0 |
| GPU | NVIDIA Hopper (H800) |

The model reports `model_type == "qwen3_5_moe"`: 40 layers with mixed linear/full
attention, wrapped in the Qwen3-VL multimodal architecture. That last detail causes
several of the issues below, because a text-only SFT job still traverses the multimodal
code path.

## Index

| # | Stage | Symptom | Fix |
|---|---|---|---|
| 1 | Data loading | PyArrow `offset overflow while concatenating arrays` | Split large JSON files, load from a directory |
| 2 | Data validation | `The number of images does not match the number of <image> tokens` | Override the placeholder env vars with sentinels |
| 3 | Model loading | `requires flash-linear-attention>=0.4.1` | Install `flash-linear-attention` |
| 4 | Vision-tower forward | `AttributeError: 'NoneType' object has no attribute 'to'` | One-line patch to `transformers`, or upgrade to ≥ 5.8.1 |
| 5 | Loss computation | OOM allocating ~75 GiB for `logits.float()` | Add a `qwen3_5_moe` branch to the liger dispatch table |
| 6 | Backward pass | `Triton >= 3.4.0 on Hopper GPUs produces incorrect results` | Install `tilelang` |
| 7 | Step 2 forward | `CUDA error: an illegal memory access` in the MoE block | Disable liger `swiglu`/`rms_norm`, keep only FLCE |
| 8 | Multi-node | NCCL `_REDUCE_SCATTER_BASE` watchdog timeout | Upgrade to torch ≥ 2.10 |

---

## 1. PyArrow int32 string offset overflow

**Symptom** — `datasets.Dataset.from_json` fails during `pa_table.combine_chunks()`:

```
pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays
```

**Root cause** — PyArrow's default string arrays use 32-bit offsets, capping a single
column at 2 GiB. A 2.6 GiB single-file JSON dataset overflows that limit once
`combine_chunks` merges all chunks into one string column.

**Fix** — split the file by rows into shards small enough to stay under the limit, place
them in a directory, and point `file_name` at the directory:

```
data/my_dataset_shards/
  part-00.json
  part-01.json
  part-02.json
  part-03.json
```

```json
"my_dataset": {
  "file_name": "my_dataset_shards"
}
```

`data/loader.py` calls `os.listdir` when `file_name` is a directory and picks up every
shard, so the training YAML needs no change.

**Note** — do not substitute a same-named `.jsonl` if one exists alongside; these often
carry extra `version`/`meta_info`/`tools` fields and are not content-equivalent.

---

## 2. Multimodal placeholders colliding with text content

**Symptom**

```
ValueError: The number of images does not match the number of <image> tokens
```

raised from `mm_plugin._validate_messages` → `content.count(IMAGE_PLACEHOLDER)`.

**Root cause** — `Qwen2VLPlugin._validate_messages`, inherited by `Qwen3VLPlugin`, counts
literal occurrences of `IMAGE_PLACEHOLDER` (default `<image>`) in every message. In a
text-only SFT corpus drawn from issue text, JSX, HTML and Markdown, a small fraction of
samples contain the literal string `<image>`. The plugin reads them as image
placeholders, but the batch has no images, so validation fails.

**Fix** — `extras/constants.py` already reads these from the environment, so override all
three with sentinels that cannot occur in the data. The launcher does this by default:

```bash
export IMAGE_PLACEHOLDER=${IMAGE_PLACEHOLDER:-"<|__lf_image_placeholder__|>"}
export VIDEO_PLACEHOLDER=${VIDEO_PLACEHOLDER:-"<|__lf_video_placeholder__|>"}
export AUDIO_PLACEHOLDER=${AUDIO_PLACEHOLDER:-"<|__lf_audio_placeholder__|>"}
```

No data and no source changes are required. Confirm your own corpus has zero occurrences
of the sentinels before relying on them.

---

## 3. Missing flash-linear-attention

**Symptom**

```
ImportError: Qwen3.5 packing-seq forwarding requires flash-linear-attention>=0.4.1
```

raised from `patcher.py:_check_fla_dependencies` when `model_type` is `qwen3_5` or
`qwen3_5_moe` and `flash_attn: fa2`.

**Fix** — `pip install -U "flash-linear-attention>=0.4.1"`. `install_env.sh` pins 0.5.0.

---

## 4. `s_aux=None` dereference in the transformers FA2 path

**Symptom** — `AttributeError: 'NoneType' object has no attribute 'to'` at
`transformers/integrations/flash_attention.py:84`, during the vision-tower forward.

**Root cause chain**

1. `MultiModalDataCollatorForSeq2Seq` injects a 64×64 white placeholder image into
   text-only batches (`sum(batch_imglens) == 0`). This is deliberate: it prevents the
   ZeRO-3/FSDP parameter-sync hang that occurs when the vision tower does not participate
   in the forward pass at all. See `data/collator.py`.
2. Because of that injection, the vision tower really does run, and its attention blocks
   take the FA2 path.
3. transformers 5.6.0 declares `s_aux: torch.Tensor | None = None` in
   `flash_attention_forward` but then calls `s_aux.to(query.dtype)` unconditionally.
4. `s_aux` is the learnable attention sink of the Qwen3.5 language model. The vision
   tower's attention never sets it, so the call dereferences `None`.

**Fix** — a one-line guard in site-packages, applied idempotently by `install_env.sh`:

```python
# <python-env>/site-packages/transformers/integrations/flash_attention.py
s_aux=s_aux.to(query.dtype) if s_aux is not None else None,
```

**Notes**

- Fixed upstream in transformers ≥ 5.7; 5.8.1 carries the same guard. Upgrading is the
  durable solution, but verify LlamaFactory's `patch_qwen3_5_forward` against the newer
  release first.
- Downgrading transformers is not an option: the `qwen3_5_moe` model class was only
  introduced in 5.2.0, so anything older cannot load the model at all.
- Reinstalling transformers drops the patch, which is why the installer reapplies it on
  every run.

---

## 5. Liger dispatch misses `qwen3_5_moe`, causing a loss-stage OOM

**Symptom** — a warning at startup:

```
[WARNING] llamafactory.model.model_utils.liger_kernel >> Current model does not support liger kernel.
```

followed by every rank OOM-ing in the first step's loss computation:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75-93 GiB.
  File ".../transformers/loss/loss_utils.py", line 55, in ForCausalLMLoss
    logits = logits.float()
```

**Root cause** — `enable_liger_kernel: true` silently did nothing. The dispatch table in
`src/llamafactory/model/model_utils/liger_kernel.py` covered `qwen3_5` (dense) and
`qwen3_moe` (the older Qwen3-MoE) but had no `qwen3_5_moe` branch, so it fell through to
the unsupported path. liger-kernel 0.8.0 does implement
`apply_liger_kernel_to_qwen3_5_moe`, exposed lazily under `liger_kernel.transformers`.

Without fused linear cross-entropy, a `(B=1, L=131072, V≈151936)` float32 logits tensor
needs roughly 75 GiB — an immediate OOM.

**Fix** — this repository adds the missing branch:

```python
elif model_type == "qwen3_5_moe":
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe as apply_liger_kernel
```

`apply_liger_kernel_to_qwen3_5_moe` defaults to `fused_linear_cross_entropy=True`, so the
logits are never materialized and peak memory becomes negligible.

**Verification** — startup logs should show
`Liger kernel has been applied to the model.`

---

## 6. fla refuses the gated_delta_rule backward on Hopper with Triton ≥ 3.4.0

**Symptom** — the first backward pass raises:

```
RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results
for gated chunk_bwd_dqkwg (see #640).
Please install tilelang: `pip install tilelang`
```

from `fla.ops.gated_delta_rule.chunk.backward` → `chunk_gated_delta_rule_bwd` →
`chunk_bwd_dqkwg`.

**Root cause** — fla 0.5.0 added a hard guard in `fla/ops/common/chunk_o.py`. Upstream
issue #640 reported `dg`/`dk`/`db` gradient errors of 1–5% (some absolute errors above
1.0) on Hopper with Triton 3.5, which makes training **diverge silently**. The
maintainers conservatively widened the guard to all Hopper parts with Triton ≥ 3.4.0 and
require the tilelang backend instead. A cu128 PyTorch build pins `triton==3.4.0`, so an
H800 always trips it.

**Fix**

```bash
python -m pip install tilelang
```

fla's `BackendRegistry` detects tilelang via `find_spec` and switches automatically — no
code change and no environment variable needed.

**Verification** — the fla logger prints
`[FLA Backend] common.chunk_bwd_dqkwg -> tilelang`.

**Workarounds to avoid**

- Downgrading Triton below 3.4.0 breaks the cu128 PyTorch build, which hard-depends on it.
- `FLA_DISABLE_BACKEND_DISPATCH=1` bypasses the dispatch framework and lands back on the
  same `raise`.
- Deleting the guard from fla's source lets training run, but reintroduces exactly the
  silent gradient corruption the guard exists to prevent.

**Note** — tilelang JIT-compiles its kernels, so the first backward pass has a cold-start
cost of tens of seconds to a few minutes; later steps read from the on-disk cache. Import
emits a stream of `Field "..." duplicates an ancestor field` warnings from tvm-ffi, which
are harmless.

---

## 7. Liger SwiGLU/RMSNorm swaps cause an illegal memory access on step 2

**Symptom** — step 1 trains normally, then the step 2 forward crashes:

```
File ".../transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py", line 788, in forward
    shared_expert_output = F.sigmoid(self.shared_expert_gate(...)) * shared_expert_output
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The process dies with SIGABRT. The stack trace ends in `Py_FinalizeEx` because the
destructor re-enters CUDA and aborts a second time; the real fault is at the sigmoid above.

**Root cause** — `apply_liger_kernel_to_qwen3_5_moe` does three things by default:

1. `fused_linear_cross_entropy=True` — replaces the model's `forward` with the LCE
   version. **This is the part that fixes the issue 5 OOM.**
2. `swiglu=True` — swaps the `Qwen3_5MoeExperts` class for `LigerExperts` and the
   per-layer `mlp.shared_expert` instances for `LigerQwen3MoeSwiGLUMLP`.
3. `rms_norm=True` — replaces `Qwen3_5MoeRMSNorm` with `LigerRMSNormForQwen3Next` (note:
   Qwen3-**Next**'s norm, not Qwen3.5's own implementation).

LlamaFactory calls `apply_liger_kernel(**kwargs)` without passing a model instance, so
only the *class* substitutions take effect. Those are still enough to cause damage:
rebinding the module-level `Qwen3_5MoeExperts` symbol interacts badly with LlamaFactory's
`patch_qwen3_5_forward`, which rewrites `Qwen3_5MoeDecoderLayer.forward` wholesale. The
illegal access appears after ZeRO-3's first reshard, when step 2's forward re-gathers
parameters.

CUDA reports illegal accesses asynchronously, so the sigmoid is not the offending kernel —
but the stack does place the fault inside `Qwen3_5MoeSparseMoeBlock`, with the
`LigerExperts` class swap and the `LigerRMSNormForQwen3Next` hidden-dimension mismatch as
the candidate sources. liger-kernel 0.8.0's Qwen3.5-MoE swaps were evidently not validated
against ZeRO-3 combined with LlamaFactory's patcher.

**Fix** — this repository disables the two problematic substitutions for `qwen3_5_moe`
while keeping FLCE:

```python
# src/llamafactory/model/model_utils/liger_kernel.py
if model_type == "qwen3_5_moe":
    kwargs.update({"swiglu": False, "rms_norm": False})

apply_liger_kernel(**kwargs)
```

**Notes**

- FLCE is the part that matters for memory. Liger's SwiGLU and RMSNorm have negligible
  effect on peak memory in long-context SFT, so turning them off does not bring back the
  issue 5 OOM.
- Do not respond to this crash by setting `enable_liger_kernel: false` — that reinstates
  the 75 GiB OOM.
- If upstream liger-kernel fixes ZeRO-3 compatibility for Qwen3.5-MoE, deleting the
  `kwargs.update(...)` line re-enables both.

---

## 8. Multi-node NCCL `_REDUCE_SCATTER_BASE` watchdog timeout

**Symptom** — single-node 8-GPU training is stable, but scaling to 4 nodes / 32 GPUs
fails after a few hundred steps with all ranks reporting simultaneously:

```
[rankN]:[E ProcessGroupNCCL.cpp:685] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=43410, OpType=_REDUCE_SCATTER_BASE,
           NumelIn=526336, NumelOut=16448, Timeout(ms)=600000)
  ran for 600006 milliseconds before timing out.
```

Collective state is perfectly symmetric across ranks and loss is healthy right up to the
hang, which rules out per-rank divergence or a single rank running out of memory.

**Root cause** — after an extended investigation across NCCL transports (socket-on-eth0,
RoCE v2 with GPUDirect RDMA, and socket over the RoCE bond), the deciding factor turned
out to be **PyTorch itself**. Upgrading `torch` from `2.8.0+cu128` to `2.10.0+cu128`
eliminated the hang: 4 nodes × 32 GPUs then trained continuously across epoch boundaries
without ever falling back to auto-retry.

**Fix**

```bash
python -m pip install --upgrade "torch==2.10.0"
```

Keep the rest of the stack as validated: Python 3.12 / CUDA 12.8 / Triton 3.4.0 /
transformers 5.6.0 / fla 0.5.0 / liger-kernel 0.8.0.

**Likely mechanism** (not bisected; 2.9.0 was never tested in isolation)

- **Bundled NCCL version.** torch 2.8.0 ships NCCL 2.27.x; 2.10.0 ships a newer release.
  NCCL 2.28+ fixed several collective hang and desync bugs. Check with
  `python -c "import torch; print(torch.cuda.nccl.version())"`.
- **ProcessGroupNCCL rework.** The `WorkNCCL` enqueue/complete/abort paths changed between
  2.9 and 2.10, which may have closed a race.
- **Flight Recorder changes.** 2.10 deprecated `TORCH_NCCL_TRACE_BUFFER_SIZE` in favour of
  `TORCH_FR_BUFFER_SIZE`, implying internal restructuring.

If you must stay on torch 2.8.x, start by overriding NCCL alone to the version 2.10 bundles.

### Defence in depth retained in the scripts

These settings are no longer load-bearing on torch ≥ 2.10, but they are side-effect free
and cover both flaky fabrics and a forced downgrade, so the launcher keeps them:

- **Pin the interface.** In containers with several NICs (`eth0`, `veth1`, `docker0`),
  NCCL's auto-detection can hand off to a virtual interface: the handshake succeeds and
  large cross-node messages then hang. `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` /
  `TP_SOCKET_IFNAME` are set explicitly. Adjust them to your own interface name.
- **Multi-threaded sockets.** `NCCL_SOCKET_NTHREADS=8`, `NCCL_NSOCKS_PERTHREAD=8` and an
  8 MiB `NCCL_BUFFSIZE` keep tail latency down when IB is disabled.
- **Step-level checkpoints.** `save_strategy: steps` with `save_steps: 50` bounds the work
  lost to any interruption. With epoch-level saves, a hang partway through a multi-hour
  epoch discards everything since the last boundary.
- **Auto-retry.** The launcher wraps `torchrun` in a retry loop that rescans for the newest
  checkpoint, so a watchdog abort resumes rather than ending the run.
- **Short watchdog.** `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180`. This value controls how long
  to keep waiting *after* a hang is detected, not how much delay is tolerated, so raising
  it only delays recovery. Raising it to 1800s was tested and never self-healed.

### If you need to enable RoCE

The launcher defaults to `NCCL_IB_DISABLE=1`. On a fabric where RDMA is known good, enable
it and let NCCL pick the HCA nearest each GPU by PCIe affinity:

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=<comma-separated HCA list>
export NCCL_IB_GID_INDEX=3       # standard RoCE v2 IPv4 layout; verify on your nodes
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=1
```

Confirm the transport actually came up by grepping the first few hundred lines of NCCL
logs for `NET/IB : Using [0]<hca>:1/RoCE` and
`GPU Direct RDMA (nvidia-peermem) enabled`. If you instead see `Using network Socket`,
something is still forcing the socket path.

Leave `NCCL_IB_TC` and `NCCL_IB_SL` at their defaults unless your cluster operators supply
values — writing an SL that the switch has no matching PFC priority for causes packet loss
and a new class of hang.

### Reading the fabric without rdma-core tools

Containers rarely ship `ibstat` / `show_gids`, but if `/dev/infiniband` is passed through,
sysfs has everything:

```bash
# Link state and link layer (Ethernet => RoCE, InfiniBand => true IB)
for d in /sys/class/infiniband/*; do
  for p in "$d"/ports/*; do
    echo "$(basename "$d") port=$(basename "$p") \
layer=$(cat "$p/link_layer" 2>/dev/null) \
state=$(cat "$p/state" 2>/dev/null) \
phys=$(cat "$p/phys_state" 2>/dev/null) \
rate=$(cat "$p/rate" 2>/dev/null)"
  done
done

# HCA-to-GPU PCIe affinity: expect GPU_i and NIC_i to show as PIX
nvidia-smi topo -m

# GID table: pick the index whose type is RoCE v2 for NCCL_IB_GID_INDEX
for f in /sys/class/infiniband/*/ports/*/gid_attrs/types/*; do
  [ -s "$f" ] || continue
  port_dir=$(dirname "$(dirname "$f")")
  echo "$(basename "$(dirname "$(dirname "$port_dir")")") \
port=$(basename "$port_dir") idx=$(basename "$f") type=$(cat "$f")"
done
```

Expect `state` to read `4: ACTIVE` and `phys_state` to read `5: LinkUp`. A GID table
typically lays out RoCE v1 at even indices and RoCE v2 at odd ones, but read it rather
than assuming — the correct `NCCL_IB_GID_INDEX` is whichever index reports RoCE v2.

---

## Harmless warnings

- `NCCL INFO NET/Plugin: Failed to find ncclNetPlugin_v10 symbol` — NCCL 2.27's native ABI
  is v10 while HPC-X ships a v9 RDMA plugin. It falls back automatically; the following
  `Loaded net plugin NCCL RDMA Plugin v9` line is the one that took effect.
- `Environment variable TORCH_NCCL_TRACE_BUFFER_SIZE is deprecated` — the launcher already
  uses the new `TORCH_FR_BUFFER_SIZE` name.
- `The fast path is not available because ... causal-conv1d` — fla's PyTorch fallback
  notice. Correctness is unaffected. Installing `causal-conv1d` speeds things up but adds
  ABI risk.
- `tvm_ffi/registry.py: UserWarning: Field '...' duplicates an ancestor field` — tilelang
  internal noise, see issue 6.

## Healthy-start checklist

A correctly configured run prints all of these during startup:

```
[INFO] llamafactory.model.model_utils.liger_kernel >> Liger kernel has been applied to the model.
[FLA Backend] common.chunk_bwd_dqkwg -> tilelang
```

and, on multi-node runs, NCCL selecting the interface you intended rather than a virtual
one. Loss and `grad_norm` should both decrease smoothly; a `grad_norm` that falls from
roughly 3 to around 1 over the first ten steps is normal SFT behaviour for this setup.

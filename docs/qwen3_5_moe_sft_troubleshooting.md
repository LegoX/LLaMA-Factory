# Qwen3.5-35B-A3B-Base SFT 故障排查记录

本文档记录在 `lf_v2` 环境下运行 `run_sft_qwen3_5_35b_a3b_base.sh` 全量 SFT(单机 8×H800 / DeepSpeed ZeRO-3 / FA2 / cutoff_len=131072)过程中遇到的 7 个连续报错,以及在扩展到 4 节点 32 卡时遇到的第 8 个跨机通信类报错及其修复。

## 环境基线

| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| PyTorch | **2.10.0+cu128**(原 2.8.0+cu128;多机训练问题最终由 2.8.0 → 2.10.0 升级根治,见 Bug 8j) |
| Triton | 3.4.0 |
| CUDA | 12.8 |
| transformers | 5.6.0 |
| flash-linear-attention (fla) | 0.5.0 |
| liger-kernel | 0.8.0 |
| GPU | NVIDIA Hopper (H800) ×8 |
| 模型 | Qwen3.5-35B-A3B-Base(`model_type == "qwen3_5_moe"`,40 层混合 linear/full attention,Qwen3VL 多模态包装) |

## Bug 索引

| # | 阶段 | 错误关键词 | 修复方式 |
|---|---|---|---|
| 1 | 数据加载 | PyArrow `offset overflow while concatenating arrays` | 拆分大 JSON,改用目录式 `data_files` |
| 2 | 数据校验 | `The number of images does not match the number of <image> tokens` | 用 sentinel 字符串覆盖 `IMAGE/VIDEO/AUDIO_PLACEHOLDER` 环境变量 |
| 3 | 模型加载 | `ImportError: Qwen3.5 packing-seq forwarding requires flash-linear-attention>=0.4.1` | `pip install -U "flash-linear-attention>=0.4.1"` |
| 4 | 视觉塔前向 | `AttributeError: 'NoneType' object has no attribute 'to'`(transformers FA2 `s_aux=None`) | 一行补丁 `transformers/integrations/flash_attention.py:84` |
| 5 | 训练 loss 计算 | OOM:`logits.float()` 试图分配 ~75 GiB | 在 `liger_kernel.py` dispatch 表里补 `qwen3_5_moe` 分支 |
| 6 | 反向传播 | `RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg` | `pip install tilelang` |
| 7 | step 2 forward | `CUDA error: an illegal memory access` 在 `Qwen3_5MoeSparseMoeBlock` sigmoid | dispatch 里给 qwen3_5_moe 显式 `swiglu=False, rms_norm=False`,只保留 FLCE |
| 8 | 多机训练 | NCCL `_REDUCE_SCATTER_BASE` watchdog 超时,32 卡同步卡死,且每次重启都丢光当前 epoch 的训练 | **最终解(8j):升级 torch 2.8.0 → 2.10.0,多机 hang 根治**。前置工程化兜底(8e–8f)仍保留作纵深防御:socket-on-eth0 + `save_strategy: steps / save_steps=50` + 脚本外层 auto-retry。曾尝试 patch DeepSpeed 子 PG timeout 至 1800s(8d),实测无效已撤回(8g);8h 新增 socket-on-bond 通路、8i 三通路对比锁定 fabric 嫌疑并准备升 vendor 工单——后被 8j 的 torch 升级一次性绕过,vendor 工单不再需要 |

---

## Bug 1 — PyArrow int32 字符串偏移溢出

### 现象

```
pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays
```

调用栈:`datasets.Dataset.from_json` → `pa_table.combine_chunks()`。

### 根因

PyArrow 默认字符串数组用 int32 偏移,单列总字节数 < 2 GiB。
`step_3.5_sweagent2520_oh4000_cc_5596-Pangu_s-add_info_12116.json`(11827 条记录,2.6 GiB)在 `combine_chunks` 阶段把所有 chunk 合并成单条字符串列时溢出 int32 上限。

### 修复

把单个大 JSON **按行拆成 4 份**(每份 < 2 GiB),用目录式 `file_name`:

1. 拆分:每份 2959/2959/2958/2958 条,SHA256 与原文件等价
   ```
   data/pangu_code_data/
     step_3.5_sweagent2520_oh4000_cc_5596-Pangu_s-add_info_12116_shards/
       part-00.json
       part-01.json
       part-02.json
       part-03.json
   ```
2. 改 `data/dataset_info.json`:
   ```json
   "step_3.5_sweagent2520_oh4000_cc_5596-Pangu_s-add_info_12116": {
     "file_name": "pangu_code_data/step_3.5_sweagent2520_oh4000_cc_5596-Pangu_s-add_info_12116_shards"
   }
   ```

LlamaFactory `data/loader.py:74-80` 在 `file_name` 为目录时会自动 `os.listdir` 收齐所有分片;YAML 不需要改。

### 备注

- **不要**用同名 `.jsonl` 替代:它含额外 `version`/`meta_info`/`tools` 字段,与 `.json` 内容不等价。
- 后续新数据集若 > 2 GiB,直接按这个目录形式准备即可。

---

## Bug 2 — 多模态 placeholder 与文本内容冲突

### 现象

```
ValueError: The number of images does not match the number of <image> tokens
```

调用栈:`mm_plugin._validate_messages` → `content.count(IMAGE_PLACEHOLDER)`。

### 根因

LlamaFactory 的 `Qwen2VLPlugin._validate_messages`(被 `Qwen3VLPlugin` 继承)对每条文本统计 `IMAGE_PLACEHOLDER` 字面子串的出现次数,默认值是 `<image>`。

40/16580 条训练样本(纯文本 SFT,数据来自 issue 描述、JSX、HTML、Markdown)里恰好包含 `<image>` 字面字符串,被 plugin 当成图片占位符,但 batch 里实际没有图片 → 校验失败。

### 修复

利用 `extras/constants.py` 已有的 `os.getenv("IMAGE_PLACEHOLDER", "<image>")` 机制,在启动脚本里把三种占位符**全部覆盖为数据中不可能出现的 sentinel**。`run_sft_qwen3_5_35b_a3b_base.sh` 第 53 行附近已加:

```bash
export IMAGE_PLACEHOLDER=${IMAGE_PLACEHOLDER:-"<|__lf_image_placeholder__|>"}
export VIDEO_PLACEHOLDER=${VIDEO_PLACEHOLDER:-"<|__lf_video_placeholder__|>"}
export AUDIO_PLACEHOLDER=${AUDIO_PLACEHOLDER:-"<|__lf_audio_placeholder__|>"}
```

### 备注

- 三种数据集都已确认零碰撞(grep 过 `<|__lf_image_placeholder__|>` 等)。
- **不修改任何数据,不修改任何源代码**,仅环境变量。
- 纯文本 SFT 命中此问题的本质原因:Qwen3.5-35B-A3B-Base 是多模态-capable base,LlamaFactory 仍走 mm 路径。

---

## Bug 3 — flash-linear-attention 缺失

### 现象

```
ImportError: Qwen3.5 packing-seq forwarding requires flash-linear-attention>=0.4.1
```

调用栈:`patcher.py:_check_fla_dependencies`(在 `model_type ∈ {qwen3_5, qwen3_5_moe}` 且 `flash_attn=fa2` 时触发)。

### 根因

`lf_v2` 环境没装 fla。Qwen3.5 的 linear-attention 层(混合 32 个 GatedDeltaNet linear-attention + 8 个 full-attention)依赖 `fla.modules.convolution.causal_conv1d` 和 `fla.ops.gated_delta_rule.{chunk,fused_recurrent}_gated_delta_rule`。

### 修复

```bash
python -m pip install -U "flash-linear-attention>=0.4.1"
```

实际装入 `flash_linear_attention 0.5.0` + `fla_core 0.5.0`。

---

## Bug 4 — transformers 5.6.0 FA2 `s_aux=None` 解引用

### 现象

```
AttributeError: 'NoneType' object has no attribute 'to'
```

位置:`transformers/integrations/flash_attention.py:84`,出现在视觉塔前向。

### 根因链

1. `MultiModalDataCollatorForSeq2Seq` 在纯文本 batch(`sum(batch_imglens)==0`)时注入 64×64 白图占位符,**目的是避免 ZeRO-3/FSDP 在没有图片时 vision tower 不参与 forward 导致的参数同步 hang**(见 `data/collator.py:343-358`)。
2. 注入后 vision tower 实际跑 forward,其 attention 块通过 FA2 路径。
3. transformers 5.6.0 的 `flash_attention_forward` 函数签名声明 `s_aux: torch.Tensor | None = None`,但函数体里第 84 行**无条件**调 `s_aux.to(query.dtype)`。
4. `s_aux` 是 Qwen3.5 LLM 主体的 learnable attention sink,vision tower 的 attention 没设这个参数 → `None.to(...)` 抛异常。

### 修复

最小一行补丁:

```python
# <python-env>/site-packages/transformers/integrations/flash_attention.py:84
s_aux=s_aux.to(query.dtype) if s_aux is not None else None,  # FA only accepts half precision
```

### 备注

- 此 bug 已在 transformers ≥ 5.7 / 5.8.1 上游修复(`s_aux=(s_aux.to(query.dtype) if s_aux is not None else None)`)。
- **transformers 4.50.0 不能用作降级方案** —— `qwen3_5_moe` 模型类直到 5.2.0 才被引入(PR #43830 "Adding Support for Qwen3.5"),4.50.0 上模型连加载都会失败。
- 若环境重装 transformers,此补丁会丢;长期方案是升级到 transformers ≥ 5.8.1,但需先验证 LlamaFactory 的 `patch_qwen3_5_forward` 与新版兼容。

---

## Bug 5 — Liger kernel dispatch 漏 `qwen3_5_moe` → loss OOM

### 现象

启动日志中先出现:
```
[WARNING] llamafactory.model.model_utils.liger_kernel:149 >> Current model does not support liger kernel.
```
紧接训练第一步在 loss 计算处 8 卡同时 OOM:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75-93 GiB.
  ...
  File ".../transformers/loss/loss_utils.py", line 55, in ForCausalLMLoss
    logits = logits.float()
```

### 根因

YAML 配的 `enable_liger_kernel: true` 没生效:

- `src/llamafactory/model/model_utils/liger_kernel.py` 的 dispatch 表里只有 `qwen3_5`(dense)、`qwen3_moe`(老的 Qwen3-MoE),**没有 `qwen3_5_moe` 分支** → fallback 到"不支持"路径。
- liger-kernel 0.8.0 实际**已实现** `apply_liger_kernel_to_qwen3_5_moe`(`monkey_patch.py:3055`),通过懒加载暴露在 `liger_kernel.transformers` 命名空间。
- 没启用 fused linear cross-entropy 的后果:`(B=1, L=131072, V≈151936) float32` logits 张量需要 ~75 GiB,直接 OOM。

### 修复

在 `src/llamafactory/model/model_utils/liger_kernel.py:84` 附近增加分支:

```python
elif model_type == "qwen3_5":
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5 as apply_liger_kernel
elif model_type == "qwen3_5_moe":
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe as apply_liger_kernel
elif model_type == "gpt_oss":
    ...
```

`apply_liger_kernel_to_qwen3_5_moe` 默认 `fused_linear_cross_entropy=True`,logits 不再被物化,内存峰值从 ~75 GiB 降至可忽略。

### 验证

启动日志应出现:
```
[INFO] llamafactory.model.model_utils.liger_kernel:144 >> Liger kernel has been applied to the model.
```

---

## Bug 6 — fla 在 Hopper + Triton ≥ 3.4.0 上拒绝运行 gated_delta_rule 反向

### 现象

反向传播第一步:
```
RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results
for gated chunk_bwd_dqkwg (see #640).
Please install tilelang: `pip install tilelang`
```

调用栈:`fla.ops.gated_delta_rule.chunk.backward` → `chunk_gated_delta_rule_bwd` → `chunk_bwd_dqkwg`。

### 根因

fla 0.5.0(PR #827)在 `fla/ops/common/chunk_o.py:715-720` 加了硬性 guard:

```python
if g is not None and IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0:
    raise RuntimeError(
        "Triton >= 3.4.0 on Hopper GPUs produces incorrect results for "
        "gated chunk_bwd_dqkwg (see #640). Please install tilelang: "
        "`pip install tilelang`"
    )
```

issue #640 报告 H20 + Triton 3.5 上 `dg`/`db`/`dk` 反向梯度精度异常(error ratio 1-5%,部分绝对误差 >1.0,会让训练**静默发散**)。维护者保守地把 guard 范围扩到 Hopper(H100/H800/H20)+ Triton ≥ 3.4.0,要求换到 tilelang 后端。

环境正好命中:`torch 2.8.0+cu128` 把 `triton==3.4.0` 钉死,GPU 是 H800 → 必须走 tilelang。

### 修复

```bash
python -m pip install tilelang
```

实际装入 `tilelang 0.1.9` + `apache-tvm-ffi 0.1.11` + `cloudpickle 3.1.2` + `ml-dtypes 0.5.4` + `torch-c-dlpack-ext 0.1.5` + `z3-solver 4.15.4.0`,无版本冲突。

fla 的 `BackendRegistry`(`fla/ops/backends/__init__.py`)会通过 `find_spec("tilelang")` 自动检测并启用 tilelang 后端,**不需要改代码**,也不需要设环境变量(`FLA_TILELANG=1` 在 Hopper + Triton ≥ 3.4.0 上默认开启)。

### 验证

启动后 fla logger 出现:
```
[FLA Backend] common.chunk_bwd_dqkwg -> tilelang
```

### 不应采用的"绕过"方案

- ❌ **降级 Triton 到 < 3.4.0**:torch 2.8.0+cu128 硬依赖 triton==3.4.0,会破坏 PyTorch。
- ❌ **设 `FLA_DISABLE_BACKEND_DISPATCH=1`**:会跳过整个 dispatch 框架,直接撞回原 raise。
- ❌ **手改 fla 源码删 raise**:guard 是为了防止**静默精度错位**(issue #640 实测 dg diff 1.29、训练发散),删了能跑但模型质量不可信。

### 备注

- tilelang 是 JIT 编译的 TVM 派生方案,**首次反向会编译 kernel,有 cold-start 延迟**(几十秒到数分钟),后续 step 从 cache 读取(默认 `~/.tvm_ffi_cache/` 或 `~/.cache/tilelang/`)。
- import 时会有大量 `Field "..." duplicates an ancestor field` 警告,是 tvm-ffi 内部 noise,可忽略。
- PR #827 同时修了一个 Hopper 专属的二次精度 bug:tilelang kernel 里 `T.copy(shared→fragment)-as-consumer` 的写法在 Blackwell 上能跑但在 Hopper 上会 race。最终修复后 H100 上 dg diff 从 1.29 降到 0.004。

---

## Bug 7 — Liger SwiGLU/RMSNorm 替换在 step 2 forward 触发 illegal memory access

### 现象

第一个 step **训练正常**(`loss=0.4572, grad_norm=2.918`),但进入第二个 step 的 forward 时崩溃:

```
File ".../transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py", line 788, in forward
    shared_expert_output = F.sigmoid(self.shared_expert_gate(...)) * shared_expert_output
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

进程被 SIGABRT 终止(exitcode = -6),Stack trace 顶端是 Python 解释器 `Py_FinalizeEx`(因为 destructor 时再次调 CUDA 触发二次 abort,真正的根因在前面 sigmoid 这一步)。

### 根因

`apply_liger_kernel_to_qwen3_5_moe` 在默认参数下做三件事:

1. `fused_linear_cross_entropy=True`:替换 `Qwen3_5MoeForConditionalGeneration.forward` 为 LCE 版本(**这是真正解决 Bug 5 OOM 的部分**)
2. `swiglu=True`:把 `Qwen3_5MoeExperts` 类整个换成 `LigerExperts`,把每个 decoder layer 的 `mlp.shared_expert` 实例换成 `LigerQwen3MoeSwiGLUMLP`,把 `mlp.experts` 实例换成 `LigerExperts`
3. `rms_norm=True`:把 `Qwen3_5MoeRMSNorm` 替换为 `LigerRMSNormForQwen3Next`(注意是 Qwen3-**Next** 的 norm,不是 Qwen3.5 自己的 norm 实现)

LF 当前用 `apply_liger_kernel(**kwargs)` 调用(不传 `model` 实例),所以只有 **类替换** 生效,instance 替换在 LF 这种调用方式下是 noop。但 **类替换仍会污染**:`Qwen3_5MoeExperts = LigerExperts` 改了模块级符号绑定,而 LF 的 `patch_qwen3_5_forward` 又重写了整个 `Qwen3_5MoeDecoderLayer.forward`,两者的交叠在 ZeRO-3 第一次 reshard 之后(step 2 fwd 重新 all-gather 参数时)出现非法访存。

时序证据:
- step 1 fwd ✅(fla forward via Triton + LF patcher 正常)
- step 1 bwd ✅(tilelang JIT 编译 8 个 kernel,~12 秒,bwd 通过)
- step 1 optimizer ✅(`loss=0.4572`、`grad_norm=2.918` 正常打印)
- step 2 fwd ❌(进入 `_patched_decoder_forward` → `self.mlp(...)` → sigmoid 越界)

> CUDA "illegal memory access" 是异步报告的,sigmoid 不是真正出错的 kernel。但 stack 表明是在 `Qwen3_5MoeSparseMoeBlock` 里发生的,候选越界源是 LigerExperts 类替换 / LigerRMSNormForQwen3Next 用错了 hidden 维度。

Liger 0.8.0 对 Qwen3.5-MoE 的 swiglu/rms_norm 替换显然没在 ZeRO-3 + LF patcher 这条路径上充分验证。

### 修复

LF 的 dispatcher 里给 `qwen3_5_moe` 显式关掉 swiglu / rms_norm,**只保留 FLCE**(那才是解决 Bug 5 OOM 的部分):

```python
# src/llamafactory/model/model_utils/liger_kernel.py
if require_logits and "fused_linear_cross_entropy" in inspect.signature(apply_liger_kernel).parameters:
    logger.info_rank0("Current training stage does not support chunked cross entropy.")
    kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}
else:
    kwargs = {}

if model_type == "qwen3_5_moe":
    # Liger's swiglu/rms_norm class swaps for Qwen3.5-MoE cause an illegal memory
    # access on step 2 forward under ZeRO-3 (observed on H800 + transformers 5.6.0
    # + liger-kernel 0.8.0). Keep only fused_linear_cross_entropy, which is the
    # part that actually resolves the long-context loss-stage OOM.
    kwargs.update({"swiglu": False, "rms_norm": False})

apply_liger_kernel(**kwargs)
```

### 验证

冒烟测试通过,启动后日志应仍显示:

```
[INFO] llamafactory.model.model_utils.liger_kernel:144 >> Liger kernel has been applied to the model.
```

但内部只替换 LM head + loss 路径,不改 SwiGLU 和 RMSNorm。

### 备注

- FLCE 是真正必需的部分(它把 `(B=1, L=131072, V≈151936) float32` logits 物化避免掉,直接节省 ~75 GiB)。Liger 的 SwiGLU/RMSNorm 在长上下文 SFT 上对显存影响可忽略,关掉不会重新触发 Bug 5 的 OOM。
- 如果上游 Liger 修了对 Qwen3.5-MoE 的 ZeRO-3 兼容性,可以重新打开:把 `kwargs.update({"swiglu": False, "rms_norm": False})` 这行删除即可。
- 不应直接关掉整个 Liger(`enable_liger_kernel: false`),那会回到 Bug 5 的 75 GiB OOM。

---

## Bug 8 — 4 节点 NCCL `_REDUCE_SCATTER_BASE` 600 s watchdog 超时

### 现象

单机 8 卡稳定后,把训练扩展到 4 节点 32 卡,跑到第 ~69 步(后修一次部分参数后能跑到第 ~388 步)时,**所有 32 个 rank 同时**报:

```
[rankN]:[E ProcessGroupNCCL.cpp:685] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=43410, OpType=_REDUCE_SCATTER_BASE,
           NumelIn=526336, NumelOut=16448, Timeout(ms)=600000)
  ran for 600006 milliseconds before timing out.
[rankN] PG status: last enqueued work: 43412, last completed work: 43409
...
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: src/train.py FAILED
[traceback : Signal 6 (SIGABRT) received]
```

四份 node log:

```
logs/train_20260516_004703_node0.log    # 第 1 次,69 步崩
logs/train_20260516_004732_node1.log
logs/train_20260516_004740_node2.log
logs/train_20260516_004747_node3.log
logs/train_20260516_084852_node0.log    # 第 2 次,388 步崩
logs/train_20260516_084920_node1.log
logs/train_20260516_084929_node2.log
logs/train_20260516_084936_node3.log
```

### 根因

三个层面问题叠加:

1. **NCCL 自动选错网卡(第 1 次 69 步崩的根因)**
   容器里有 `eth0`(管理面)、`veth1`(容器虚拟网卡)、`bonding_masters` 等多张网卡,NCCL 默认会按一套启发式扫描,在多网卡环境下经常握手到 `veth1`/`docker0`/`lo`——握手能过,但跨机大消息一定挂。脚本注释自己也说了"多网卡环境建议额外指定 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`",但运行时一直没设。

2. **没用上 RoCE,跨机数据面只剩一根 200G eth0 的纯 TCP socket(第 2 次 388 步崩的根因)**
   节点其实有 8 张 ConnectX RoCE bond(`/sys/class/infiniband/mlx5_bond_0..7`,Ethernet/200 Gb/sec/Active),与 8 张 GPU 在 PCIe 拓扑上 PIX 一一对应,但脚本没启用 IB 路径,所有 32 卡的 ZeRO-3 reduce-scatter 全部塞进单根 eth0 + 单 socket 流。MTU=1500、单线程 socket、无 RDMA,任何一次 TCP 抖动都让某个大消息卡住,集合通信全员等同步,10 分钟后被 watchdog 杀掉。

3. **ZeRO-3 子进程组 watchdog 默认 600 s,且 yaml 里的 `ddp_timeout` 不作用于子 PG**
   日志里 `PG ID 1` 就是 DeepSpeed 在初始化 ZeRO-3 时另开的子组,这个子组的 `Timeout(ms)=600000` 写死,不继承 `TrainingArguments.ddp_timeout=180000000`。再大的 yaml `ddp_timeout` 都救不了它,只能通过环境变量层面的 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` 覆盖。

集合通信状态完全对齐(每个 rank `last_enqueued=243750 / last_completed=243747 / stuck on 243748`)、loss 在崩溃前一直健康下降——可以排除"某 rank 计算分歧"或"某 rank 显存爆掉",纯粹是底层网络挂死。

### 排查方法(无 ibstat / show_gids 工具时)

容器里通常**不会预装 rdma-core 用户态工具**(`ibstat`、`rdma`、`show_gids`、`sminfo` 全部缺失),但只要 `/dev/infiniband` 透传进来了,所有信息都能从 sysfs 读出来。在每个计算节点上跑:

```bash
# 1) 链路状态 + link layer (RoCE 还是真 IB)
for d in /sys/class/infiniband/*; do
  hca=$(basename "$d")
  for p in "$d"/ports/*; do
    port=$(basename "$p")
    state=$(cat "$p/state" 2>/dev/null)        # 期望: "4: ACTIVE"
    pstate=$(cat "$p/phys_state" 2>/dev/null)  # 期望: "5: LinkUp"
    rate=$(cat "$p/rate" 2>/dev/null)
    layer=$(cat "$p/link_layer" 2>/dev/null)   # InfiniBand 或 Ethernet
    echo "$hca port=$port  layer=$layer  state=$state  phys=$pstate  rate=$rate"
  done
done

# 2) HCA 与 GPU 的 PCIe 亲和: 看 nvidia-smi topo -m, 期望 GPU_i 与 NIC_i 是 PIX
nvidia-smi topo -m

# 3) GID 表(选 RoCE v2 那一行的 idx 作为 NCCL_IB_GID_INDEX)
for f in /sys/class/infiniband/*/ports/*/gid_attrs/types/*; do
  idx=$(basename "$f"); type=$(cat "$f" 2>/dev/null)
  port_dir=$(dirname $(dirname "$f"))
  hca=$(basename $(dirname $(dirname "$port_dir")))
  port=$(basename "$port_dir")
  [ -z "$type" ] && continue
  echo "$hca port=$port idx=$idx type=$type"
done
```

示例节点探测结论(原始探测输出不纳入仓库):

| 项 | 实测 |
|---|---|
| `/dev/infiniband` | 存在,`uverbs0..8` + `umad*` + `rdma_cm` 齐全 |
| HCA | `mlx5_bond_0..7` 共 8 张 |
| link_layer | **Ethernet**(即 RoCE,不是真 IB) |
| state / phys_state | `4: ACTIVE` / `5: LinkUp` |
| rate | `200 Gb/sec (2X NDR)` |
| 后端 netdev | `reth0/2/4/6/8/10/12/14`(纯 RDMA,无 IPv4) |
| GID 表布局 | `idx=0,2 = IB/RoCE v1`;`idx=1,3 = RoCE v2` |
| GPU↔NIC 亲和 | GPU_i 与 NIC_i 全是 `PIX`(同 PCIe Switch) |

→ **该选 `NCCL_IB_GID_INDEX=3`(RoCE v2 IPv4 布局),HCA 列表写满 8 张让 NCCL 按 PIX 自动分配**。

### 修复

`run_sft_qwen3_5_35b_a3b_base.sh` 的 `export` 段加入两组变量(已落盘):

```bash
# OOB(进程组建立、torchrun rendezvous、gloo)走 eth0; 数据面走 RoCE。
# 多网卡环境若 NCCL 自动探测错网卡(走到 docker0/veth1/lo)极易导致跨机集合通信 hang/超时。
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}

# 跨机数据面走 RoCE v2: 节点有 8 个 mlx5_bond_(0..7) HCA,与 GPU PIX 一一对应,
# Ethernet/200 Gb/sec/Active. GID 表 idx=3 是标准 RoCE v2 IPv4 布局。
# 写满 8 个 HCA 让 NCCL 按 PIX 亲和给每张 GPU 选最近的 bond。
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
# 大消息(ZeRO-3 切片)在 RoCE 上多 QP 并行,扛拥塞、压平尾延迟。
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
export NCCL_IB_SPLIT_DATA_ON_QPS=${NCCL_IB_SPLIT_DATA_ON_QPS:-1}

# watchdog/诊断: ZeRO-3 子 PG 默认 600s 太紧,且 yaml 的 ddp_timeout 不作用于子 PG。
# 下次再 hang 能 dump 出实际卡住的 collective + 落后 rank。
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}
export TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE:-2097152}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
```

该配置与常见多节点脚本中 `NCCL_SOCKET_IFNAME=eth0 / GLOO_SOCKET_IFNAME=eth0` 的写法一致。

### 易踩坑(避免回退)

修这个 bug 中途**曾把脚本写成"socket-only"配方**(`NCCL_NET=Socket` / `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD` / `NCCL_BUFFSIZE`)。这套是**机器无 IB 时**的兜底方案,会强制走 TCP,在有 RoCE 的环境下反而把 RoCE 关掉。最终 commit 已**完整移除**这些参数,只保留上面那一组。如果以后看到日志里 `Using network Socket` 而不是 `NCCL RDMA Plugin`,先检查脚本里有没有人误加 `NCCL_NET=Socket` / `NCCL_IB_DISABLE=1`。

`NCCL_IB_TC` / `NCCL_IB_SL`(交换机 PFC/QoS 相关)默认值即可,**不要瞎设**——硬写错值(交换机 priority 没配 3 时强写 SL=3)会丢包,反而触发新的 hang。集群同事如果给过推荐值,按他们的来。

### 验证

跑 2 节点,头 200 行 NCCL 日志(`logs/train_20260516_171027_node0.log`)出现下列关键行,即代表 RoCE 通路完整建立:

```
NCCL version 2.27.3+cuda12.9
NET/Plugin: Loaded net plugin NCCL RDMA Plugin v9 (v9)
Plugin Path : /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
NET/IB : Made virtual device [0..7] name=mlx5_bond_0..7 speed=200000 ndevs=1
NET/IB : Using [0]mlx5_bond_0:1/RoCE [1]mlx5_bond_1:1/RoCE ... [7]mlx5_bond_7:1/RoCE [RO]; OOB eth0:<private-ip><0>
NET/IB : GPU Direct RDMA (nvidia-peermem) enabled for HCA 0 'mlx5_bond_0
NET/IB : GPU Direct RDMA (DMABUF) enabled for HCA 0 'mlx5_bond_0
P2P plugin v9 IBext_v9
NVLS multicast support is available on dev N (NVLS_NCHANNELS 16)
Channel 03/0 : 11[3] -> 3[3] [send] via NET/NCCL RDMA Plugin v9/3/GDRDMA
Connected all trees
```

要点:
- `Using [0..7] mlx5_bond_*:1/RoCE` —— 8 张 HCA 全部进入 NCCL 通信器
- `GPU Direct RDMA (nvidia-peermem) enabled` 与 `(DMABUF) enabled` —— GDR 两条路径都通,HCA 直接读写 GPU 显存,**完全绕开 CPU**
- `Channel ... via NET/NCCL RDMA Plugin v9/3/GDRDMA` —— rank 11(node1 GPU3) ↔ rank 3(node0 GPU3) 走 mlx5_bond_3,PIX 亲和正确
- `OOB eth0:...` —— 控制面如期走 eth0,不在数据面竞争

修复后 2 节点 16 卡 step 时长(`{'loss': '0.4572', ...}` 起):

| step | 时间(s) | 与 4 节点 socket 旧记录对比 |
|---|---|---|
| 1 | 133 | 旧:237 |
| 2 | 122 | 旧:195 |
| 3 | 113 | 旧:175 |
| 4 | 108 | 旧:165 |

**卡数减半反而更快**,数学上前 4 步 loss 与单机 8 卡 baseline 完全一致(`0.4572 / 0.4675 / 0.4903 / 0.4672`),说明 RoCE 启用没有改变训练数值,纯粹消除了通信瓶颈。

### 8c 后续:4 节点首次 RoCE 跑挂在 step 271(瞬态拥塞,非配置失效)

`logs/train_20260516_173749_node{0..3}.log`,从 checkpoint-260 续训,17:43 进训练,17:56 进入第 7420 号 collective(`_REDUCE_SCATTER_BASE`),所有 32 个 rank 卡死,18:06 全部 600 s watchdog 超时。

观察 PG 状态(每个 rank 都是):
```
last enqueued work: 7422, last completed work: 7419
[0, 1, ..., 31] joined but didn't finish collective #7422
```

**32/32 完美对称**——不是某个 rank 落后导致的 desync,也不是 init 失败回退。RoCE 通路这把已经建立(对照 8b 同位置直接初始化失败)、step 261~271 走完正常(单步 ~63 s,与 2 节点验证同量级),说明这是**RoCE 跑久了某条 QP 上一段时间内重传超过 IB 重试预算,把整个 ring 拖死**的瞬态拥塞,不是配置问题。

针对性收紧:
- `NCCL_IB_TIMEOUT=22 → 23`:单次 CQ 重试 timeout 从 ~17.7 s 抬到 ~35 s,扛过更长瞬态拥塞。
- `NCCL_IB_QPS_PER_CONNECTION=4 → 2`:bond LAG 下多 QP 容易把 5-tuple hash 撞到同一 leg,反而制造 elephant-flow 级拥塞;2 QP 已够压尾延迟。
- `NCCL_IB_PCI_RELAXED_ORDERING=1`:GPUDirect RDMA 走 PCIe 时允许 RO,缓解 ConnectX 对 P2P write 排队的反压。
- `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH`:**临时**调到 INFO,下次再 hang 时能从日志反推走的哪条 HCA / 哪段 ring/tree。日志会涨,稳定后再调回 WARN。

不动的:
- `NCCL_IB_RETRY_CNT=7` 已是上限。
- DeepSpeed ZeRO-3 子 PG 的 600 s watchdog 是 `dist.new_group(ranks)` 没传 timeout 用了 torch 默认值,**不读** `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`(后者只对默认 PG 有效)。8c 这一步先靠减小拥塞概率扛过去,8d 实测不够,见下节。

### 8d 后续:8c 调参后 step 286 又同样超时,改 patch DeepSpeed

`logs/train_20260516_182635_node{0..3}.log`,把 8c 的 `IB_TIMEOUT=23 / QPS=2 / PCI_RELAXED_ORDERING=1` 落进脚本后再跑一次 4 节点。INFO 级别的 NCCL 日志这次能看到完整初始化:

```
NET/IB : Using [0..7]mlx5_bond_*:1/RoCE [RO]; OOB eth0:<private-ip><0>
NET/IB : GPU Direct RDMA (nvidia-peermem) enabled
NET/IB : GPU Direct RDMA (DMABUF) enabled
ncclCommInitRankConfig ... rank 0..7 nranks 32 - Init COMPLETE
Connected all trees
```

8 张 HCA 全部进入,`[RO]` 表示 PCI Relaxed Ordering 已生效,nranks=32 通信器一切正常。

但 step 286(单步 ~50 s,与 8c 同量级)之后,所有 32 个 rank 又卡死在 `_REDUCE_SCATTER_BASE` #16855,600 s 后集体 watchdog,状态 **`last enqueued=16857 / last completed=16854`**——与 8c 同形态。

意义上的判断:
- 8c 的 IB 重试预算扩到 ~245 s(`IB_TIMEOUT=23 × IB_RETRY_CNT=7`)仍然没救回,collective 600 s 才被 watchdog fire,说明这个网络上的瞬态停顿是**分钟级**的。常见原因:交换机 PFC 没对 DSCP/PCP 配齐、ECN 标记后没走 CC、incast 把交换机出口队列打死等,均不在训练机这一层能修。
- 8c 把 failure 从 step 271 推到 step 286(只多撑 15 步),说明 NCCL 层任何调参只能提升"安全间隔",不能根治。
- 整段 RoCE init 干净,GPU 与 HCA PIX 亲和正确,这把不是配置失效,是网络真撑不住。

#### 真正的解法:patch DeepSpeed 子 PG watchdog

`yaml ddp_timeout` 只作用于默认 PG;`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` 也只看默认 PG。DeepSpeed ZeRO-3 创建的所有子 PG 走 `dist.new_group(ranks)`,**没传 timeout**,torch 默认 `timedelta(seconds=600)`,这个 600 s 是 NCCL `WorkNCCL` watchdog 真正用的那个值。要改只能改源码或 patch:

```python
# <python-env>/site-packages/deepspeed/comm/torch.py:373

# 改前:
def new_group(self, ranks):
    return torch.distributed.new_group(ranks)

# 改后(读 env DEEPSPEED_PG_TIMEOUT_SEC, 默认 1800s):
def new_group(self, ranks):
    import os
    from datetime import timedelta
    sec = int(os.environ.get("DEEPSPEED_PG_TIMEOUT_SEC", "1800"))
    return torch.distributed.new_group(ranks, timeout=timedelta(seconds=sec))
```

脚本同步增加 `export DEEPSPEED_PG_TIMEOUT_SEC=1800`(已落入 `run_sft_qwen3_5_35b_a3b_base.sh`)。

含义:
- 给训练 30 min 而不是 10 min 去扛过 RoCE 网络抖动;真正死了的 collective 还是会被 fire,只是阈值放宽。
- 配合 `RESUME=1` 自动续训,即便最坏情况(单次 hang 真挂了 30 min),也能从最近 checkpoint 重启而不是手工介入。
- patch 写在 site-packages 里(本仓库不动 deepspeed 源码),换 env 重装包要重 apply;在 `整体修复清单` 里也补了对应一行。

#### 不再尝试的方向

- `NCCL_IB_TIMEOUT/RETRY_CNT` 继续往上抬:已经接近 IB 协议上限,再抬只是徒增 false-positive 之间的间隔。
- 关 `NCCL_IB_QPS_PER_CONNECTION`(降到 1):2 → 1 损失多 QP 抗拥塞,8c 现象表明 QP 数不是主因。
- 改 `ddp_timeout`:对 ZeRO-3 子 PG 无效,前面验证过。

### 8e 最终方案:回退到 socket-on-eth0

把 8a~8d 四次 run 横向比一下:

| run | 配置 | MTBF | 撑了多少步 | 平均步耗 |
|---|---|---|---|---|
| 8b (`20260516_084852`) | 纯 eth0 socket(无 RoCE) | **6 h 44 min** | 388 步 | ~62 s/step |
| 8c (`20260516_173749`) | RoCE + QPS=4 | ~12 min | 11 步(resume 后) | ~63 s/step |
| 8d (`20260516_182635`) | RoCE + QPS=2 + RO | ~12 min | 26 步 | ~62 s/step |

**步耗几乎一样,但 RoCE 的 MTBF 从 6h44m 掉到 12min**——折腾 RoCE 在这个 fabric 上是负收益。

为什么反直觉:
- TCP 单次重传能拖到几分钟,kernel 自己擦屁股,NCCL socket 这一层根本感知不到瞬态丢包。
- RDMA 的 IB QP 重试预算是 `IB_TIMEOUT × IB_RETRY_CNT ≈ 245 s`,超了就 fatal、QP 死掉、NCCL 永远等不到 ACK,watchdog 600 s fire。
- 本机房 fabric 上的瞬态停顿是分钟级(PFC/ECN 没配齐),刚好踩在 TCP 能扛、RDMA 扛不住的区间。
- ZeRO-3 + gradient checkpointing 的算/通比下,RoCE 的带宽优势在 backward 里被算力盖掉,实测每步差不到 1 s。

最终落入脚本的 NCCL 段:

```bash
export NCCL_DEBUG=WARN

# OOB + 数据面统一走 eth0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

# 明确禁掉 IB 探测,不让 NCCL 再尝试走 mlx5_bond_*
export NCCL_IB_DISABLE=1
# socket 模式下并行收发线程,压尾延迟。8 × 8 = 64 路 socket 已足够喂满 100/200G eth0
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8

# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 只影响默认 PG,无副作用,留着
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# DEEPSPEED_PG_TIMEOUT_SEC 在 8g 已撤回(配合 8d patch 一起回滚),不再 export
```

8c 加的 `NCCL_IB_TIMEOUT / NCCL_IB_QPS_PER_CONNECTION / NCCL_IB_SPLIT_DATA_ON_QPS / NCCL_IB_PCI_RELAXED_ORDERING / NCCL_IB_HCA / NCCL_IB_GID_INDEX` 已**全部删除**——`NCCL_IB_DISABLE=1` 之后这些都没意义,留着只会误导后人。

8d 的 `deepspeed/comm/torch.py:new_group` patch **保留**——跟走不走 RoCE 无关,纯粹给 ZeRO-3 子 PG 一个更宽容的 watchdog 阈值,无副作用。**(更正:8g 撤回了这个 patch,见下文)**

### 8f 真正的问题不在网络,在 checkpoint 间隔

`logs/train_20260516_194508_node{0..3}.log`,8e 落定后再跑 4 节点。1800s patch 实测生效(日志里 `Timeout(ms)=1800000`),socket-on-eth0 也按预期建立。**227 步 / ~3h51m** 后照样 hang,32 rank 完美对称卡 #142792。

但跨 5 次 4 节点 run 横向看,真正的痛点冒出来了:

| run | 配置 | wall-clock | 步数 | 累计跑到 step | 丢失 |
|---|---|---|---|---|---|
| 8b | eth0 socket | 6h44m | 388 | 388(从 0 开始) | 128 步(epoch 边界 step 260 之后) |
| 8c | RoCE QPS=4 | ~12 min | 11 | 271(resume 260) | 11 步 |
| 8d | RoCE QPS=2 | ~12 min | 26 | 286(resume 260) | 26 步 |
| **8e** | socket+patch | **3h51m** | **227** | **487**(resume 260) | **227 步** |

8e 比 8b/8c/8d 都稳——但**`saves/...` 目录里仍然只有 `checkpoint-260` 一个**。

#### 根因

yaml `save_strategy: epoch`,而每 epoch=260 步 ≈ 4h22min。本机房 fabric MTBF 大概在 **4–7h**——单 epoch 跨度刚好略大于 MTBF,**每一次都跑到 epoch 中段就 hang**,然后 resume 的时候发现没有比 step 260 更新的 checkpoint,**全部退回到上一个 epoch 边界**。

8e 跑了 227 步、写了 ~3h45min 的梯度,只差 33 步就到 step 520 的下一个 epoch 边界——一次都没保住。

#### 修复:把 checkpoint 频率追上 MTBF

```yaml
# examples/train_full/qwen3_5_35b_a3b_base_..._epo3.yaml
save_strategy: steps
save_steps: 50           # ~50min 一次,单次 hang 顶多丢 ~50min
save_total_limit: 3      # 35B + ZeRO-3 优化器状态单 ckpt ~400GB,限磁盘
```

50 步对应 ~50 min,跟单次 ckpt 写盘开销(35B + opt state 几百 GB,~3-5 min)的比值约 5-10%,可接受。

#### 修复:脚本外层加 auto-retry 循环

watchdog 触发后整个 torchrun 退出。脚本里已经有 `RESUME=1` 自动从最近 checkpoint 续训,但需要人手再敲一次 `bash run_sft_...sh`。把 `torchrun ...` 抽成 `run_torchrun()`,外面套:

```bash
attempt=0
while : ; do
    attempt=$((attempt + 1))
    run_torchrun; rc=$?
    [ "$rc" -eq 0 ] && exit 0
    [ "$AUTO_RETRY" != "1" ] && exit "$rc"
    [ "$attempt" -ge "$MAX_RETRIES" ] && exit "$rc"
    clear_gpu_procs
    sleep "$RETRY_BACKOFF"
    rescan_resume          # 重扫 checkpoint-* 取最新的喂回 resume_from_checkpoint
    TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"
done
```

环境变量:
- `AUTO_RETRY=1`(默认开),`AUTO_RETRY=0` 走老语义(挂了就不重试)
- `MAX_RETRIES=20`(默认),防止 checkpoint 没动还在死循环
- `RETRY_BACKOFF=60`(默认),给集体故障 + GPU 清理留 60s 缓冲

每次重试会:
1. `clear_gpu_procs` 清掉 zombie 推理 / NCCL 残留进程
2. `sleep RETRY_BACKOFF`
3. `rescan_resume` 重扫 `output_dir/checkpoint-*`,把最新的 step 喂回 `EXTRA_ARGS`
4. 重开一个新的 `train_<ts>_node<r>.log`(避免覆盖前一次的诊断日志)

#### 期望效果

按 50min checkpoint + 4-7h MTBF 估算:
- 单次 hang 平均损失 ~25 min(50min 间隔的一半);
- 780 步 × 1min ≈ 13h 总训练时长,期间预计 hang 2-3 次;
- 总浪费时间 ~50-75 min,是 8e(单次就丢 3h45min)的 1/4 量级。

#### 不做什么

- 不再调 NCCL/IB 任何参数——8c/8d/8e 已经穷举,网络层面无法根治。
- 不去改交换机 PFC/ECN——不在训练机权限范围内,且不掌握集群拓扑。

### 回归注意

- 一次"全 rank 完美对称卡 #N、`last_enqueued = #N+2 / last_completed = #N-1`"在 RoCE 上是**瞬态拥塞**的标准 signature,不是 desync——**不要去找"哪个 rank 卡了"**,真正的着力点是网络抖动是否能被吸收。本机房选的是 socket+TCP 让 kernel 重传吸收,见 8e。
- `NCCL_DEBUG` 最终回到 `WARN`(8c 临时调到 INFO 是为了抓 RoCE 走的具体 HCA;8e 已经放弃 RoCE,WARN 即可)。再排查时手动 `NCCL_DEBUG=INFO bash run_sft_...sh` 即可。
- 8d patch 必须在每个节点的 deepspeed site-packages 里都打过;`pip install -U deepspeed` / 重建镜像后要重 apply,否则 `dist.new_group(ranks)` 会回到 600 s 默认。**(8g 已撤回 patch,这条历史保留)**
- 若后续想再尝试 RoCE,先确认机房交换机 PFC/ECN 已经按 RoCE v2 配齐(DSCP→PCP 映射、ECN 标记、CC 算法),否则会重复 8c/8d 的剧本。
- 不要用 yaml `ddp_timeout` 来"延长容忍"——那只作用于默认 PG;真正影响 ZeRO-3 子 PG 的是 `dist.new_group(ranks, timeout=...)`,见 8d patch。

### 8g 撤回 8d 的 deepspeed patch

8d 把 ZeRO-3 子 PG watchdog 从 600s 抬到 1800s,设计初衷是给 TCP 重传更多自愈窗口。8e 跑了一把 4 节点(`logs/train_20260516_194508_*`),日志确认 patch 生效(`Timeout(ms)=1800000`)。**但 collective 跑满整个 1800s 也没自愈**——这次 hang 跟 600s 默认 watchdog 下的 hang 没本质区别,只是单纯多等了 1200s 才让 watchdog fire。

横向对比:

| run | watchdog | wall-clock | 步数 | 注 |
|---|---|---|---|---|
| 8b | 600s 默认 | **6h44m** | 388 | 没 patch |
| 8e | 1800s patched | 3h51m | 227 | 1800s 全等满,没救回来 |

考虑:
- N=1 vs N=1,样本量不足以说明 patch 让 MTBF 下降——但**也没证据 patch 让 MTBF 上升**。
- 边际收益分析: Linux TCP 默认 `tcp_retries2≈15min` 后会强制断连。能在 600~900s 之间靠 TCP 自愈的窗口本来就窄,加上集群级 incast / PFC 抖动通常超 15min,patch 命中的概率偏低。
- 边际成本是确定的: 每次 hang 多浪费 1200s 才能进 auto-retry。
- 8f 已经把 checkpoint 间隔压到 50 步(~50min),单次 hang 损失上限已被 yaml 控住,patch 那 1200s 反而是负担。

所以撤回 patch:

```python
# <python-env>/site-packages/deepspeed/comm/torch.py:373
# 恢复成 vanilla:
def new_group(self, ranks):
    return torch.distributed.new_group(ranks)
```

脚本里 `DEEPSPEED_PG_TIMEOUT_SEC` 不再需要,删掉。`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` 保留——它只影响默认 PG,无副作用。

#### 何时考虑把 patch 加回来

如果 auto-retry 后频繁出现"重启没几步又 hang"的链式失败(说明 TCP 那一波**本来在恢复**但被 600s 太早 fire 了),再把 8d patch 加回来,默认值可以调成 1200s(取 600s 和 1800s 的中间值)。在那之前先吃 vanilla 配置的简洁性。

### 8h 第三条通路:socket-on-RoCE-bond(绕开 virtio eth0)

8e 的 socket-on-eth0 在 5/16 撑了 6h44m / 388 步,5/17 退化到 ~25min,**配置零变更**。这种"网络维度退化但端侧没动"的情况,8b/8c/8d/8e 在两条通路(verbs RoCE / kernel TCP eth0)都已经验证过——但其实容器里**还有第三条通路一直没用**。

#### 拓扑发现

在训练节点容器内跑:

```bash
ls /sys/class/infiniband/*/device/net   # mlx5 → netdev 映射
for d in /sys/class/net/*; do
    n=$(basename "$d"); drv=$(readlink "$d/device/driver" 2>/dev/null | xargs -r basename)
    sp=$(cat "$d/speed" 2>/dev/null); ms=$(readlink "$d/master" 2>/dev/null | xargs -r basename)
    printf "%-12s drv=%-12s speed=%-7s master=%s\n" "$n" "${drv:-?}" "${sp:-?}" "${ms:-}"
done
```

输出:

```
bond0..bond7    drv=?            speed=400000  master=
reth0..reth15   drv=mlx5_core    speed=200000  master=bondN
eth0            drv=virtio_net   speed=200000  master=
mlx5_bond_0 -> reth0
mlx5_bond_1 -> reth2
...
mlx5_bond_7 -> reth14
```

关键:

- **`eth0` 是 `virtio_net`**(virtio 软栈),不是物理网卡——之前 8e 默认走的"socket-on-eth0"实际是过虚拟化层
- **`bond0..bond7` 是真实 mlx5 物理 netdev**(8 根 LACP bond,每根 200G × 2 = 400Gbps),容器里**完整暴露了 kernel netdev** 而不仅仅是 verbs 设备
- 8 根 bond 在节点上各有独立 IPv4(具体地址已脱敏),属于典型 multi-rail RoCE 设计

这意味着除了 8c/8d 走过的 verbs/RDMA 路径、8e 走的 virtio TCP 路径之外,**还有第三条没试过**:**mlx5 物理路径 + kernel TCP 栈**(`NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=bond0..bond7`)。它跟前两条都不重合:

| 维度 | eth0 socket | RoCE verbs | bond socket(8h 新) |
|---|---|---|---|
| 物理路径 | virtio 软栈 | mlx5 物理 | **mlx5 物理** |
| 协议栈 | kernel TCP | verbs/RDMA | **kernel TCP** |
| 流控 | TCP 拥塞控制(CUBIC) | PFC + ECN(易死锁) | **TCP 拥塞控制** |
| 带宽上限 | 200G 标称 | 8 × 200G | **8 × 400G**(multi-rail) |
| MTBF 历史 | 5/16=6h44m → 5/17=25min | ~12min | 待测 |

#### 假设

5/17 之后退化的极可能不是物理 fabric,而是 **virtio eth0 这条软路径**——管理面共享、MTU 1500、走 host kernel 转发,任何邻居容器/管理面流量打满都会传染。bond0..7 是直通 SR-IOV 物理 NIC,跟 virtio 不共享数据通路;且 RoCE PFC 死锁是 verbs 层的问题,kernel TCP 不踩。

如果假设成立,socket-on-bond 应当**比 8e 稳,且比 8c/8d 稳**——既绕开 virtio,又绕开 PFC。

#### 实施

`run_sft_qwen3_5_35b_a3b_base.sh` 里把通路选择抽成 `NCCL_TRANSPORT` 开关,默认值 `eth0`(保持 8e 行为),新增 `bond` 模式。切换:

```bash
NCCL_TRANSPORT=bond NNODES=4 NODE_RANK=$R bash run_sft_qwen3_5_35b_a3b_base.sh
```

`bond` 模式自动设:

```bash
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7
GLOO_SOCKET_IFNAME=bond0      # rdzv 单 rail 够用
NCCL_SOCKET_NTHREADS=8
NCCL_NSOCKS_PERTHREAD=8       # 64 路并行 socket 喂满 8 × 400G
```

同时脚本启动期会调用 `dump_net_topo()`,把 bond/IB/route/fib_trie 状态写进 train log,后续排查 fabric 退化不需要再登节点跑 sysfs。

#### 验收

- bond 模式下 NCCL 启动 log 应出现 `NET/Socket : Using [bond0:..., bond1:..., ...]`(而不是 `[eth0:...]`)
- 单步耗时:期望 ≤ socket-on-eth0(50-62s/step);因为 multi-rail 并行,大概率更快
- MTBF:跑满一轮 4h+ 不挂为成功;如仍 ~25min hang,**则 5/17 退化确实是物理 fabric/spine 问题**,落到 vendor escalation(见 `qwen3_5_moe_sft_cluster_vendor_escalation.md`)

#### 副作用 / 注意

- 8 根 bond 必须每根都有 IP **且跨节点同子网内可路由**——若供应商只给 bond0 配了 IP,NCCL 会 fallback 到单 rail(等于纯 eth0 替换为 bond0,仍是物理路径但只 400G,不是 8 × 400G)
- bond 模式不影响 RDMA 库(verbs)的对外能力,只是 NCCL 不再用;若同节点其他进程仍在跑 RDMA,互不冲突
- `NCCL_TRANSPORT=roce` 模式保留作为对照,但 8c/8d 已实证 PFC 死锁,**不要长期用**

#### 8h sysfs 探测后的细节修正(2026-05-17 晚)

`fabric_test.sh` 在训练节点跑出来后纠正了几个之前的盲点:

1. **GID v2 索引不是 3,而是 1**(`/sys/class/infiniband/mlx5_bond_*/ports/1/gid_attrs/types/1` 的内容是 `RoCE v2`)。`run_sft_qwen3_5_35b_a3b_base.sh` 的 `roce` 分支默认值已改为 `NCCL_IB_GID_INDEX=1`。8c/8d 之前 hang 的一个可能解释就是 idx=3 选到了非 v2 或非 global 的 GID,跨节点根本没建立 QP——下次再试 RoCE 时这条要单独验证。
2. **bond 是 LACP `layer3+4` 哈希**,单 TCP 流只走一个 200G slave,要靠 NCCL 多 socket(`NCCL_NSOCKS_PERTHREAD=8 × NCCL_SOCKET_NTHREADS=8 = 64 流`)才能把 2 个 slave 都铺满 → 已在 `bond` 分支默认设好。
3. **GPU/NIC NUMA 完美对齐**(NUMA 0: GPU 4 张 + reth0..7;NUMA 1: GPU 4 张 + reth8..15),走 bond 时每张 GPU 用同 NUMA 的 NIC,不跨 socket QPI;走 eth0 (virtio) 时所有数据汇到一个虚拟队列,丢失 NUMA locality。
4. **sysfs `/sys/class/net/<dev>/qos/pfc_enable` 读不到**,容器内**无法独立验证 PFC 是否真启用**,这块只能让供应商从交换机侧 dump counter(对应 vendor escalation 工单第 2 项)。
5. fib_trie LOCAL 段里的大量 K8s ClusterIP(kube-proxy 注入)并非本机 IP;真实节点地址已脱敏。8 根 bond 分别位于独立子网,确认是 multi-rail 设计。
6. **eth0 是机器的默认出口**(网关地址已脱敏),管理面/K8s/外网/容器内 DNS 全走它。这意味着 `socket-on-eth0` 模式下 NCCL 流量与所有管理面共享同一根 virtio 软栈;**邻居容器/host 任何高流量任务都会传染**,这是配置零变更但 MTBF 退化的最大嫌疑根因——不是物理 fabric 烂,是 virtio 共享路径被压垮。
7. **每根 bond 使用独立子网和 gateway**(地址与路由 metric 已脱敏),意味着各 bond 对接不同的 ToR/leaf 交换机,fabric 侧物理上是多路 ECMP。kernel 默认 metric 选 bond0,但 NCCL 用 `SO_BINDTODEVICE` 显式绑每个 socket 到指定 bond,绕过 metric 选择,所以 `NCCL_SOCKET_IFNAME=bond0,...,bond7` 真正起作用。
8. **/sys 上 `mlx5_ib / ib_core / ib_uverbs / rdma_cm` 的 `version` 文件缺失但 RDMA 栈是好的**:`/sys/class/infiniband/mlx5_bond_*` 8 个 device 都在,这要求上述模块全加载。`version` 文件没有大概率是编译进 in-tree kernel 或 strip 过 modinfo,**不要被 `<not loaded>` 误导**。

### 8i 三通路对比与 fabric 升级判定(2026-05-17 晚)

8h 上线后立刻在 4 节点上跑了 `bond` 与 `roce` 两个分支,加上之前 `eth0` 历史成绩,**首次拿到三条独立通路在同一 fabric 同一时段的对比数据**。

#### 三通路 vs MTBF

| 通路 | 协议栈 | 步耗时 | 完成步数 | hang 形态 | 失败 collective |
|---|---|---|---|---|---|
| eth0 (5/16 baseline 8b) | virtio + host kernel TCP | 50-62 s | 388 (6h44m) | 全 rank join, 600s timeout | `_REDUCE_SCATTER_BASE` |
| eth0 (5/17 退化) | 同上 | 同上 | ~? (~25min) | 同上 | 同上 |
| **bond** (`run 20260517_211439`) | mlx5 SR-IOV 直通 + 容器 kernel TCP | 73-104 s 单调恶化 | **3 (~4 min)** | 全 rank join, 600s timeout | `_REDUCE_SCATTER_BASE #2077` |
| **roce** (`run 20260517_215752`) | mlx5 verbs + RoCE v2 (GID idx=3, 已纠正 8h 错误) | 52-104 s 渐增 | **10 (~21 min)** | 全 rank join, 600s timeout | `_REDUCE_SCATTER_BASE #6793` |

**三条通路均 hang 在 ZeRO-3 backward 的 `_REDUCE_SCATTER_BASE`**,32 个 rank 全 issue 同一 op、无一完成。bond 模式最快死(4 min),RoCE 略慢(21 min),eth0 历史最久(数小时)。

#### 8i 关键修正:8h GID 推断错误

8h 探测 sysfs 时只读了 `gid_attrs/types/<idx>`,看到 idx=1 的 type 字段是 `RoCE v2` 就把 `roce` 分支默认改成 `NCCL_IB_GID_INDEX=1`,导致 `run 20260517_214925` 直接报 `ibv_modify_qp -> Network is unreachable (101)` 退出。

实际上同一张 mlx5_bond_X HCA 的 sysfs 在 RoCE v2 type 下有**两条** GID:

| idx | type | gid 内容 | 含义 |
|---|---|---|---|
| 0 | IB/RoCE v1 | `fe80::...`(link-local IPv6,基于 MAC) | 过时,不用 |
| 1 | **RoCE v2** | `fe80::...`(link-local IPv6) | type 对了但跨 /29 子网**不可路由** |
| 2 | IB/RoCE v1 | `::ffff:c815:....`(IPv4-mapped) | 过时,不用 |
| **3** | **RoCE v2** | `::ffff:c815:....`(IPv4-mapped,后 4 段 hex 是 bond IPv4) | **跨 spine 路由的全局 GID,正解** |

把 mlx5_bond_0..7 的 idx=3 GID 反解码后,可与 bond0..7 的 IPv4 一一对应(具体 GID 与 IP 已脱敏),从而确认该索引正确。

修正后 `run_sft_qwen3_5_35b_a3b_base.sh` 的 `roce` 分支恢复为 `NCCL_IB_GID_INDEX=3`(也就是这家集群的历史默认值,8h 那次"纠正"是误判)。

**读 GID 的正确姿势是双字段联合** —— 不能只读 type:

```bash
for i in $(ls /sys/class/infiniband/mlx5_bond_0/ports/1/gids); do
  g=$(cat /sys/class/infiniband/mlx5_bond_0/ports/1/gids/$i)
  t=$(cat /sys/class/infiniband/mlx5_bond_0/ports/1/gid_attrs/types/$i)
  printf "idx=%s type=%s gid=%s\n" "$i" "$t" "$g"
done | grep "RoCE v2" | grep -v "fe80:"
# 留下来那条的 idx 才是 NCCL_IB_GID_INDEX 的正确值
```

#### bond 步耗时单调恶化的可能机制

bond 模式 3 步内步耗时从 103s 涨到 84s 又涨,reduce_scatter 累积更多排队张量,最终某个 op 跨过 600s 阈值。配置上 `NCCL_NSOCKS_PERTHREAD=8 × NCCL_SOCKET_NTHREADS=8 = 64 socket/rank-pair`,32 个 rank 全网约 31000 条长 TCP 连接,平均每根 bond ~3900 条。LACP layer3+4 哈希在这种规模下任何一条 socket 卡住都会让 collective 整批等不齐——**socket 数越多,失败入口越多**。8i 没有时间做"bond 单 rail" / "bond 低 socket"对照,但这个嫌疑写下来,真要再调 bond 时先试 `NCCL_NSOCKS_PERTHREAD=2 NCCL_SOCKET_NTHREADS=2`。

#### 升 vendor 工单的判定:三独立协议栈同形态 hang = 物理 fabric 问题

写这条作为 vendor escalation 文档的 smoking gun:

- **virtio_net eth0 + host kernel TCP**(管理面虚拟接口)
- **mlx5_core SR-IOV bond0..7 + 容器内 kernel TCP**(物理 NIC,绕开 host)
- **mlx5_bond_0..7 verbs/RDMA RoCE v2**(GID idx=3 IPv4-mapped,逐项验证过)

三条**完全独立的协议栈**(virtio 走 host kernel, bond 走容器内 mlx5 socket, RoCE 走 verbs 完全 bypass kernel),**均在 30 分钟内出现完全一致的 NCCL `_REDUCE_SCATTER_BASE` 600s timeout**(全 32 ranks join 同一 op、无一完成)。

三条栈的唯一公共项 = 物理 fabric(交换机 / spine / 端口 / 光纤)。所以问题不在 NCCL 配置、不在 deepspeed、不在 PyTorch、不在 GID/QP/PFC env,**就在 fabric**。

请供应商 dump:
- 每根 bond 对端 ToR 端口的 PFC counter / pause frame rate / CRC error / discard counter
- 节点间 spine 8 路 ECMP 的 path utilization 与丢包率
- 故障时段所有 ToR/spine 的 link flap、BER、optical power 日志

#### 行动项

- [x] 修正 `run_sft_qwen3_5_35b_a3b_base.sh` `roce` 分支 `NCCL_IB_GID_INDEX=3`,撤销 8h 错改
- [x] 修正 `docs/qwen3_5_moe_sft_network_transport_guide.md` 第 2 节 / 第 5 节 / 第 6.3 节 GID 索引说明
- [x] ~~按上述 smoking gun 升 vendor 工单~~ — **8j 升级 torch 2.10.0 后 hang 不复现,工单不再需要**
- [x] ~~在等供应商回复期间,回退到 8e 的 `eth0` 通路 + auto-retry + save_steps=50 把现有训练继续推进~~ — **由 8j 取代**

### 8j 最终解:升级 torch 2.8.0 → 2.10.0,多机 hang 根治

8a–8i 整段排查的工作假设是"网络层 fabric / NCCL 配置层"出问题。最终验证下来,**根因其实在 torch 自身**:把 `torch` 从 `2.8.0+cu128` 升级到 `2.10.0+cu128` 之后,4 节点 32 卡训练不再复现 `_REDUCE_SCATTER_BASE` watchdog 超时,可以稳定连续训练而不依赖 auto-retry。

#### 操作

```bash
python -m pip install --upgrade "torch==2.10.0"
# 同步带上 torch 生态(确认 transformers/deepspeed/fla/liger 与 2.10 兼容,
# 必要时一并升级)
```

升级后保留的环境:Python 3.12 / CUDA 12.8 / Triton 3.4.0 / transformers 5.6.0 / fla 0.5.0 / liger-kernel 0.8.0,其它修复(Bug 1–7)以及 8e socket-on-eth0 + 8f save_steps=50 + auto-retry 循环全部保留作纵深防御。

#### 验证

- 4 节点 32 卡可连续训练,跨多个 epoch 边界无 `_REDUCE_SCATTER_BASE` 600s 超时
- 单步耗时与单机 8 卡 baseline 比例正常,loss / grad_norm 数值与 8 卡 baseline 同步
- auto-retry 没有被触发(因为没再 hang)

#### 可能的机制(待进一步验证)

torch 2.10.0 相对 2.8.0 在分布式栈上有若干变化,可能与本次 hang 直接相关的方向(按怀疑度排序):

- **bundled NCCL 版本变化**: torch 2.8.0 内嵌 NCCL 2.27.x,torch 2.10.0 内嵌更高版本(`python -c "import torch; print(torch.cuda.nccl.version())"` 可确认)。NCCL 2.28+ 修了多起 collective hang / desync 相关 bug
- **ProcessGroupNCCL watchdog / work tracking 重构**: 2.9–2.10 期间 PyTorch 改了 `WorkNCCL` 的入队 / 完成 / abort 路径,可能解掉某种 race
- **Flight Recorder / FR buffer 行为变化**: 2.10 把 `TORCH_NCCL_TRACE_BUFFER_SIZE` 正式 deprecate 改为 `TORCH_FR_BUFFER_SIZE`(8 节"无害告警"已记),也意味着内部结构有所重写

具体哪条是根因,8j 暂未做 bisect(`torch==2.9.0` 没单独验证)。如果未来需要在不能升到 2.10 的环境复现这个修复,可以从 NCCL 版本入手:试 `NCCL_VERSION` 单独覆盖到 torch 2.10 同款 NCCL,看是否单独够用。

#### 8a–8i 的现状

- **保留作为故障排查史**: 整段记录了在没有 8j 这条退路时,如何在工程层把"4–7h MTBF + 单 epoch 4h22m"的死循环逼到 ~50min 损失上限,以及如何用三通路对比把问题边界压缩到 fabric。这段思路对未来"torch 升级路径被堵 + 同样 fabric 抖动"的场景仍然有用
- **保留代码层兜底**: 8e socket-on-eth0、8f save_steps=50 + auto-retry、`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` 不取消,理由是无副作用且未来若 torch 又回退就直接顶上
- **8d 的 deepspeed patch 不复活**: 8g 撤回的判断在 2.10 时代仍成立——子 PG 用 vanilla 即可
- **vendor 工单不再需要**: 8i 行动项已勾掉;但 8i 的 GID v2 索引读法(双字段联合,idx=3 才对)和拓扑笔记仍有参考价值,保留

### 回归注意(8j 后)

- 重装 torch 或重建镜像后**必须保留 torch ≥ 2.10**;若被迫回退到 2.8.x,Bug 8 整段大概率会原样复发,届时回到 8e+8f 的兜底方案(socket-on-eth0 + save_steps=50 + auto-retry)
- 升级 torch 时连带升级了 NCCL,部分 NCCL 环境变量行为可能变化;首次跑 4 节点要确认健康标志(见"启动健康标志"小节)依旧全部出现
- 不要把 8a–8i 的修复方案(`NCCL_IB_DISABLE`、`NCCL_SOCKET_NTHREADS`、auto-retry 等)在 2.10 时代全删——它们零副作用,留着对 fabric 抖动的偶发场景仍是兜底

### 无害告警(可忽略)

- `NCCL INFO NET/Plugin: Failed to find ncclNetPlugin_v10 symbol` —— NCCL 2.27 原生 ABI 是 v10,HPC-X 装的 RDMA plugin 是 v9。自动 fallback 到 v9,紧接着的 `Loaded net plugin NCCL RDMA Plugin v9` 才是真正生效的版本。
- `[W ...] Environment variable TORCH_NCCL_TRACE_BUFFER_SIZE is deprecated; use TORCH_FR_BUFFER_SIZE instead` —— PyTorch 2.x 改了 flight recorder 的环境变量名。脚本里现在已经用新名 `TORCH_FR_BUFFER_SIZE`。

---

## 整体修复清单(可复现)

按本文档跑一遍以下操作即可复现稳定状态:

```bash
# ---- 数据 ----
# (Bug 1) 把超 2GB 的 JSON 拆成 ≤4 份放入 _shards 目录,
#         在 data/dataset_info.json 里把 file_name 指到目录

# ---- 环境 ----
PIP="python -m pip"

# (Bug 8j ★ 多机训练根治) torch 升级到 2.10 — 这是 Bug 8 整段真正的解;
#                          升完后 8a-8i 的网络层兜底全部退化为"无副作用的纵深防御"
$PIP install --upgrade "torch==2.10.0"

# (Bug 3) fla
$PIP install -U "flash-linear-attention>=0.4.1"

# (Bug 6) tilelang
$PIP install tilelang

# ---- 源码补丁 ----
# (Bug 4) transformers FA2 s_aux None guard:
#   src/transformers/integrations/flash_attention.py:84
#     - s_aux=s_aux.to(query.dtype),
#     + s_aux=s_aux.to(query.dtype) if s_aux is not None else None,

# (Bug 5) LlamaFactory liger dispatch:
#   src/llamafactory/model/model_utils/liger_kernel.py 在 qwen3_5 分支后加:
#     elif model_type == "qwen3_5_moe":
#         from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe as apply_liger_kernel

# (Bug 7) 同文件,在 apply_liger_kernel(**kwargs) 之前加:
#     if model_type == "qwen3_5_moe":
#         kwargs.update({"swiglu": False, "rms_norm": False})

# ---- 启动脚本 ----
# (Bug 2) run_sft_qwen3_5_35b_a3b_base.sh 已含 sentinel 占位符 export
# (Bug 8j) 多机 hang 真正的解是 torch 2.10(见上面 "PIP install --upgrade torch==2.10.0").
#         下面 8e/8f 的脚本&yaml 修改在 2.10 时代是"无副作用纵深防御",保留即可,
#         若未来被迫回退到 torch 2.8.x 这套兜底就会立刻派上用场.
# (Bug 8e) 同脚本 export 段最终方案: NCCL_IB_DISABLE=1 +
#         NCCL_SOCKET_IFNAME=eth0 / GLOO_SOCKET_IFNAME=eth0 / TP_SOCKET_IFNAME=eth0 +
#         NCCL_SOCKET_NTHREADS=8 / NCCL_NSOCKS_PERTHREAD=8 +
#         TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# (Bug 8d/8g) deepspeed 子 PG patch 已撤回——8g 复盘 8e 实测 1800s 也没自愈,patch 边际为负.
#   保留 vanilla:
#     <python-env>/site-packages/deepspeed/comm/torch.py:373
#     def new_group(self, ranks):
#         return torch.distributed.new_group(ranks)
#   未来若发现 auto-retry 后频繁链式 hang,可重新加回(默认 1200s 而非 1800s).
# (Bug 8f) yaml 改 step 级 checkpoint + 脚本 auto-retry 循环:
#   examples/train_full/qwen3_5_35b_a3b_base_..._epo3.yaml
#     - save_strategy: epoch
#     + save_strategy: steps
#     + save_steps: 50
#     + save_total_limit: 3
#   run_sft_qwen3_5_35b_a3b_base.sh 末尾的 torchrun 抽成 run_torchrun() + while 循环
#   AUTO_RETRY=1 / MAX_RETRIES=20 / RETRY_BACKOFF=60 控制重试行为
```

启动:

```bash
cd /path/to/LLaMA-Factory

# 单机 8 卡
bash run_sft_qwen3_5_35b_a3b_base.sh

# 多机(以 4 节点为例): 每台机器跑一次,NODE_RANK 各取 0..3
NNODES=4 NODE_RANK=0 bash run_sft_qwen3_5_35b_a3b_base.sh
NNODES=4 NODE_RANK=1 bash run_sft_qwen3_5_35b_a3b_base.sh
NNODES=4 NODE_RANK=2 bash run_sft_qwen3_5_35b_a3b_base.sh
NNODES=4 NODE_RANK=3 bash run_sft_qwen3_5_35b_a3b_base.sh
```

启动健康标志(全部应出现):

| 阶段 | 期望日志 |
|---|---|
| Liger | `Liger kernel has been applied to the model.`(且只跑 FLCE,不再做 SwiGLU/RMSNorm 替换) |
| fla | `[FLA Backend] common.chunk_bwd_dqkwg -> tilelang`(首次反向时) |
| TileLang | `TileLang completes to compile kernel`(8 个 kernel,首步反向时) |
| 训练 | 第一个 step 完成 `loss=...` 出现,**第二个 step 也能进入(无 illegal memory access)** |
| 多机 NCCL(仅多节点,8e socket-only) | `NCCL INFO Bootstrap: Using eth0:...` + `NET/Socket : Using [0]eth0:...` + `Connected all trees`;**不应**再出现 `NET/IB : Using ...mlx5_bond_*` |

### 已验证稳定(`train_20260515_234544_node0.log`)

7 个修复全部生效,训练稳定进入连续步骤,数值趋势健康:

| step | loss | grad_norm | lr |
|---|---|---|---|
| 1 | 0.4572 | 2.932 | 0 |
| 2 | 0.4675 | 2.864 | 6.41e-07 |
| 3 | 0.4902 | 3.082 | 1.282e-06 |
| 4 | 0.4672 | 2.929 | 1.923e-06 |
| 5 | 0.4456 | 2.136 | 2.564e-06 |
| 6 | 0.459 | 1.462 | 3.205e-06 |
| 7 | 0.4309 | 0.7108 | 3.846e-06 |
| 8 | 0.4296 | 0.8962 | 4.487e-06 |
| 9 | 0.4438 | 1.187 | 5.128e-06 |
| 10 | 0.4221 | 0.9405 | 5.769e-06 |

- 单步约 205~310s(8×H800,gbs=64,cutoff=131072,ZeRO-3,gradient checkpointing),780 步 ETA ~52 小时。
- step 2 干净通过,Bug 7 不再复现。
- grad_norm 从 ~2.9 自然降到 ~1,loss 缓慢下降,是健康的 SFT 动力学。

### 无害告警(可忽略)

- `[transformers] The fast path is not available because one of the required library is not installed. ... causal-conv1d`:fla 的 `causal_conv1d` 未安装时的 PyTorch 回退提示,不影响正确性。如要进一步提速可装 `causal-conv1d`,但当前稳定状态下不建议动(避免引入新的 ABI 风险)。
- `tvm_ffi/registry.py:85: UserWarning: Field '...' duplicates an ancestor field.` / `[WARNING] Field "..." in type "tl.GemmSPWarpPolicy" ...`:tilelang 内部的字段重注册告警,与 Bug 6 修复同源,可忽略。

---

## 附:错误日志时间线

参考训练日志(`logs/train_<timestamp>_node{0..3}.log`),按时间顺序对应到上面 8 个 bug:

```
20260515_202304   Bug 1  PyArrow offset overflow
20260515_203553   Bug 2  <image> token mismatch
20260515_204818   Bug 3  fla missing
20260515_222335   Bug 4  s_aux=None.to()
20260515_223647   Bug 5  loss-stage OOM (liger 未生效)
20260515_225624   Bug 6  fla / Triton 3.4 guard
20260515_231218   Bug 7  step 2 fwd illegal memory access (Liger SwiGLU/RMSNorm)
20260515_234544   ✅ 单机 8 卡稳定进入连续步骤(已跑到 step 10,loss/grad_norm 正常)
20260516_004703   Bug 8a 4 节点 NCCL 自动选错网卡, step 69 集体 _REDUCE_SCATTER_BASE 600s 超时
20260516_004732           (node1 同次 run)
20260516_004740           (node2 同次 run)
20260516_004747           (node3 同次 run)
20260516_084852   Bug 8b 4 节点已锁 eth0 但仍走纯 socket, step 388 在 6h44m 后再次 600s 超时
20260516_084920           (node1 同次 run)
20260516_084929           (node2 同次 run)
20260516_084936           (node3 同次 run)
20260516_171027   ✅ 启用 RoCE v2 + GPUDirect RDMA, 2 节点 16 卡稳定进入连续步骤,
                       前 4 步 loss 与单机 baseline 完全一致, 单步 133→108s(同步骤旧值 237→165s)
20260516_173749   Bug 8c 4 节点 RoCE 已生效但 step 271 又 600s 超时, 32 rank 完美对称卡 #7420,
                       判断为 RoCE 瞬态拥塞;落 IB_TIMEOUT=23 / QPS=2 / PCI_RELAXED_ORDERING=1,
                       并临时把 NCCL_DEBUG 默认调到 INFO 以便下次抓到具体 HCA/ring 证据
20260516_173802           (node1 同次 run)
20260516_173809           (node2 同次 run)
20260516_173816           (node3 同次 run)
20260516_182635   Bug 8d 8c 调参后跑到 step 286 又 600s 超时(同形态);RoCE init 干净,
                       说明网络瞬态分钟级,NCCL 调参解决不了。patch
                       deepspeed/comm/torch.py:new_group 让子 PG timeout 读
                       DEEPSPEED_PG_TIMEOUT_SEC(脚本默认 1800s),配合 RESUME=1 自愈
20260516_182700           (node1 同次 run)
20260516_182704           (node2 同次 run)
20260516_182707           (node3 同次 run)
最终方案 8e          回退 socket-on-eth0: 横向比 8b(socket 6h44m / 388 步) vs
                       8c/8d(RoCE ~12 min),步耗几乎一致但 RoCE MTBF 掉 30+ 倍。
                       脚本删除所有 NCCL_IB_*,改 NCCL_IB_DISABLE=1 +
                       NCCL_SOCKET_NTHREADS=8 / NSOCKS_PERTHREAD=8;保留 8d patch。
20260516_194508   Bug 8f socket+patch 跑 227 步 / 3h51min 后再次 hang,但发现真正的痛点是
                       yaml save_strategy=epoch(每 epoch 260 步 ≈ 4h22min)正好踩在
                       fabric MTBF 上,每次 hang 都丢光当前 epoch。改 save_strategy=steps
                       / save_steps=50 / save_total_limit=3,并在脚本外层加 auto-retry 循环。
20260516_194539           (node1 同次 run)
20260516_194543           (node2 同次 run)
20260516_194546           (node3 同次 run)
撤回决策 8g          复盘 8b(600s 默认 watchdog,6h44m / 388 步) vs 8e(1800s patched,
                       3h51m / 227 步),N=1 vs N=1 无显著差异,但 8e 的 collective 实测
                       跑满 1800s 也没自愈,patch 那 1200s 纯属白等。撤回 8d patch,
                       deepspeed/comm/torch.py:new_group 恢复 vanilla,脚本删除
                       DEEPSPEED_PG_TIMEOUT_SEC export。依赖 600s 快速 fire +
                       auto-retry + save_steps=50 自愈。
最终解 8j ★          torch 2.8.0 → 2.10.0 升级后,4 节点 32 卡 `_REDUCE_SCATTER_BASE`
                       hang 不再复现,跨多个 epoch 边界连续训练稳定。8a-8i 全部退化为
                       无副作用纵深防御(socket-on-eth0 + save_steps=50 + auto-retry),
                       vendor 工单不再需要。
```

# Qwen3.5-35B-A3B-Base SFT 阶段级 traceback 插桩

本文档记录 2026-05-18 在 `run_sft_qwen3_5_35b_a3b_base.sh` 上加的诊断仪表,目的是回应集群供应商对 Bug 8 系列(参见 `qwen3_5_moe_sft_troubleshooting.md`)的请求:

> 你再跑一次吧,把 traceback 都打印出来,报错的时候是进行到哪一步了,是前向还是数据切片,看下是不是保存数据太长了导致的,顺便把日志打出到控制台一份。
>
> 加载数据、保存模型,前向反向推理,只要涉及最小节点都打下,看卡在哪一步。

## 背景

到 2026-05-18 为止,4 节点 32 卡训练在 socket-on-eth0 通路上仍偶发 hang。最新一次(`logs/train_20260518_051935_node0.log`):从 step 300 续训,跑到 step 605/780 后于 11:18~11:23 间静默挂死。checkpoint-600 已写出完整,日志直接断尾,**没有 Traceback / NCCL ERROR / Watchdog 行**,GPU 进程随后消失。

为什么原脚本抓不到 traceback:

1. `torchrun` 默认只把 **rank 0** 的 stdout/stderr 流到启动终端,其余 31 个 rank 的输出走 elastic agent 默认路径,事后定位困难。
2. Python 默认行缓冲;SIGABRT 来了 buffered 行就丢了。
3. `PYTHONFAULTHANDLER` 没开,SIGSEGV/SIGABRT/SIGFPE 都不会自动 dump py 栈。
4. `NCCL_DEBUG=WARN` 太静,hang 没 ERROR 就一片寂静。
5. **Trainer 内部 forward / backward / dataloader_next / save 没有任何 log**——hang 时 console 上能看到的最后一行只能告诉"上一步在 N 时刻完成",看不到当前 step 卡在哪一阶段。

## 总体方案

分两层:

- **A 层(env + torchrun):零侵入观测**——所有 rank 的输出都抓到、每行带毫秒时间戳、faulthandler 自动 dump。
- **B 层(Python monkey-patch):阶段级插桩**——`tools/trace_hook.py` 在进程启动时 patch 11 个最小阶段函数,每段开始/结束打一行 stderr。**不动训练代码**(`src/llamafactory/**`、`src/train.py`、deepspeed config、yaml 全保留)。

## 文件改动

### 新增 `tools/trace_hook.py`

启动时即生效的 monkey-patch 模块,导入即安装(模块尾部直接 `install()`)。打印目标:每次 enter / exit 都写一行 stderr,异常路径打 `phase=raise`。格式固定:

```
[trace ts=<iso毫秒> rank=<R> pid=<P> step=<N> stage=<name> phase=<enter|exit|raise> dt_ms=<...>] <extra>
```

`step` 是从 `Trainer.training_step` 入口维护的全局计数(`os.environ['RANK']` 取 rank,模块全局 `_global_step`)。`extra` 字段按 stage 不同携带 batch shape、save_dir、tensor 个数等。

被 patch 的 12 个目标(覆盖供应商问的"最小阶段"):

| 阶段 | patch 目标 | extra 字段 |
|---|---|---|
| 数据加载 (单 batch) | `torch.utils.data.dataloader._BaseDataLoaderIter.__next__` | `loader_id` |
| 数据加载 (epoch 切换) | `torch.utils.data.dataloader.DataLoader.__iter__` | `dataset_len`, `num_workers` |
| 训练 step | `transformers.Trainer.training_step` | `input_ids_shape` |
| 前向 (compute_loss) | `transformers.Trainer.compute_loss` | `input_ids_shape` |
| 反向 | `deepspeed.runtime.engine.DeepSpeedEngine.backward` | — |
| 优化器 step | `deepspeed.runtime.engine.DeepSpeedEngine.step` | — |
| Trainer 保存(总入口) | `transformers.Trainer._save_checkpoint` | `global_step`, `output_dir` |
| Trainer 模型保存 | `transformers.Trainer.save_model` | — |
| HF 权重落盘 | `transformers.modeling_utils.PreTrainedModel.save_pretrained` | `save_dir` |
| safetensors 落盘 | `safetensors.torch.save_file` | `file`, `n_tensors` |
| DeepSpeed 保存 | `deepspeed.runtime.engine.DeepSpeedEngine.save_checkpoint` | `save_dir`, `tag` |
| DeepSpeed ZeRO-3 内部 save | `deepspeed.runtime.engine.DeepSpeedEngine._save_zero_checkpoint` | — |
| 集合通信 (默认关) | `torch.distributed.barrier / all_reduce / broadcast / _reduce_scatter_base / all_gather` | — |

**为什么不直接 patch `PreTrainedModel.forward`**:`PreTrainedModel` 自己没定义 `forward`,继承的是 `nn.Module._forward_unimplemented`;真正的 `forward` 在每个具体模型类(`Qwen3_5MoeForCausalLM` 等)的 `__dict__` 里。Python MRO 直接走子类方法,patch 父类是 no-op。改 patch `Trainer.compute_loss` 后,HF `training_step` 内部 `loss = self.compute_loss(model, inputs, ...)` 这一句就是 forward 的边界。`Seq2SeqTrainer` 不重写 `compute_loss`,LlamaFactory `CustomSeq2SeqTrainer.compute_loss` 在非 ASFT 路径(本 yaml 默认)走 `super().compute_loss(...)` 命中 base 类 patch。如启用 `use_asft_loss: true` 则该路径不走 super,需另外 patch。

所有 patch 都用 `functools.wraps` + try/finally 包,**任何 patch 失败只打一条 `[trace WARN]`,不让训练挂掉**。集合通信那一组默认 `TRACE_COLLECTIVES=0` 关掉(单 step 几百次会刷屏);hang 现场再开。

冒烟验证(已通过):

```
$ RANK=0 LOCAL_RANK=0 python -c "import sys; sys.path.insert(0,'.'); import tools.trace_hook"
[trace ts=2026-05-18T11:52:40.256 rank=0 pid=... step=0 stage=trace_hook phase=installed] collectives=False
DataLoaderIter.__next__ wrapped: True
DataLoader.__iter__ wrapped: True
Trainer.training_step wrapped: True
Trainer.compute_loss wrapped: True
Trainer._save_checkpoint wrapped: True
Trainer.save_model wrapped: True
PreTrainedModel.save_pretrained wrapped: True
safetensors save_file wrapped: True
DeepSpeedEngine.backward wrapped: True
DeepSpeedEngine.step wrapped: True
DeepSpeedEngine.save_checkpoint wrapped: True
DeepSpeedEngine._save_zero_checkpoint wrapped: True
```

### 新增 `tools/trace_train.py`

10 行 runpy 包装。`TRACE_STAGES=1` 时先 `import tools.trace_hook` 触发 patch,再 `runpy.run_path('src/train.py', run_name='__main__')` 执行原入口;`TRACE_STAGES=0` 时跳过 hook(等价直跑 src/train.py)。`sys.argv[0]` 改成 train.py 的真实路径,模拟 `python src/train.py` 的语义。

### 改 `run_sft_qwen3_5_35b_a3b_base.sh`

#### 顶部 export 段(行 56 附近)

```bash
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}                   # 原值 WARN,本轮提到 INFO
# === 诊断仪表 ===
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}      # SIGABRT/SIGSEGV/SIGFPE 自动 dump py 栈
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,COLL,NET}
export TRACE_STAGES=${TRACE_STAGES:-1}                  # 阶段级插桩开关
export TRACE_COLLECTIVES=${TRACE_COLLECTIVES:-0}        # collective 级插桩(刷屏,默认关)
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-info}
```

#### `RUN_TIMESTAMP` 之后(行 217)

```bash
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-$SCRIPT_DIR/logs/nccl_trace_${RUN_TIMESTAMP}_node${NODE_RANK}_rank}"
mkdir -p "$SCRIPT_DIR/logs"
```

watchdog 触发时每 rank 生成一个 pickle:`logs/nccl_trace_<TS>_node<R>_rank<L>` —— flight recorder dump,含每个 collective 的 `last_enqueued / last_completed / desync` 状态。

#### `run_torchrun()`(行 386 起)

```bash
run_torchrun() {
    local tr_log_dir="$LOG_DIR/torchrun_${RUN_TIMESTAMP}_node${NODE_RANK}_attempt${attempt}"
    mkdir -p "$tr_log_dir"
    local entry="src/train.py"
    if [ "${TRACE_STAGES:-1}" = "1" ] && [ -f "tools/trace_train.py" ]; then
        entry="tools/trace_train.py"
    fi
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
```

变化点:

- `--tee 3`:bitmask 1=stdout、2=stderr、3=both,**所有 rank** 既前缀回流到 console 又落盘到 `logs/torchrun_*/attempt_*/<local_rank>/{stdout,stderr}.log`。可单 rank `grep '[trace]'`。
- `--redirects 3 --log-dir`:原始 stdout/stderr 落盘的目录。
- `awk -W interactive ... fflush()`:每行实时打**秒级** ISO 时间戳。**不要用 `"date +..." | getline ts; close(...)` 写法**——它每行 fork 一个 `date` 子进程,32 rank × NCCL INFO 量级下管线立刻 backlog,主日志会落后墙钟 10+ 分钟,把 hang 时刻定位扭歪。改用 `strftime(..., systime())` 单进程内置,5000 行 7.5s → 0.013s(实测 580× 加速)。毫秒精度由 trace 行内置 `ts=` 字段保留。
- `entry` 在 `TRACE_STAGES=1` + `tools/trace_train.py` 存在时才切换到 wrapper,否则退化原行为。
- `${PIPESTATUS[0]}` 仍是 torchrun 的真实 rc,awk/tee 的 0 被忽略,不破坏 auto-retry 判定。

## 用法

正常启动(默认 `TRACE_STAGES=1`):

```bash
cd /jyx_data/LLaMA-Factory-latest
NNODES=4 NODE_RANK=0 bash run_sft_qwen3_5_35b_a3b_base.sh   # node0
NNODES=4 NODE_RANK=1 bash run_sft_qwen3_5_35b_a3b_base.sh   # node1
NNODES=4 NODE_RANK=2 bash run_sft_qwen3_5_35b_a3b_base.sh   # node2
NNODES=4 NODE_RANK=3 bash run_sft_qwen3_5_35b_a3b_base.sh   # node3
```

退化为原行为(关插桩 + 关 NCCL INFO):

```bash
TRACE_STAGES=0 NCCL_DEBUG=WARN bash run_sft_qwen3_5_35b_a3b_base.sh
```

hang 现场打开 collective 级插桩(配合 NCCL flight recorder dump):

```bash
TRACE_COLLECTIVES=1 bash run_sft_qwen3_5_35b_a3b_base.sh
```

## 怎么读日志:从 trace 行定位 hang 点

每行格式:

```
2026-05-18T12:34:56 [rank=12] [trace ts=2026-05-18T12:34:56.788 rank=12 pid=98765 step=607 stage=compute_loss phase=enter] input_ids_shape=(1, 131072)
```

外层 `2026-05-18T12:34:56` 是 bash awk 加的接收时刻(秒级,单进程 strftime,不会延迟);`[rank=12]` 是 torchrun --tee 加的 rank 前缀;`[trace ...]` 内嵌 `ts=` 是 `trace_hook.py` 里的发送时刻(毫秒)。**真正的事件时刻以 trace 行内嵌 `ts=` 为准**,外层秒级 ts 只用于在主日志里粗略对齐多 rank 行序。两者差应 < 1s,如果 > 5s 说明 awk/tee 管线被 INFO 级 NCCL 日志撑爆——降 `NCCL_DEBUG=WARN` 即可。

### 找最后一个未闭合的 enter

下面这条 awk 在每个 rank 维度做 enter/exit 配对计数,留下大于 0 的就是卡住没退出的阶段:

```bash
for r in 0 1 2 3 4 5 6 7; do
    echo "=== rank $r ==="
    grep -E "rank=$r .+phase=(enter|exit|raise)" logs/train_<TS>_node*.log \
      | awk '
          { for(i=1;i<=NF;i++) if($i~/^stage=/){s=$i; break}
            if($0 ~ /phase=enter/) c[s]++
            else                   c[s]--
          }
          END { for(s in c) if(c[s]>0) print s, "imbalance=", c[s] }
        '
done
```

### 32 个 rank 同时卡的形态对照

| 形态 | 含义 | 下一步 |
|---|---|---|
| 全 rank 最后未闭合 = `compute_loss` | 卡在前向某 collective(MoE all-to-all / ZeRO-3 all-gather) | 配合 NCCL flight recorder dump 查具体 collective,落 vendor escalation 工单(`qwen3_5_moe_sft_cluster_vendor_escalation.md`) |
| 全 rank 最后未闭合 = `backward` | 反向 reduce-scatter 卡死 | 同上,vendor escalation |
| 全 rank 最后未闭合 = `dataloader_next` | 数据切片层 hang;查 input_ids.shape 是不是触到 131072 上限 | yaml 调小 `cutoff_len`、或加 `packing` |
| 全 rank 最后未闭合 = `ds_save_zero_checkpoint` / `safetensors_save_file` | **服务商怀疑的"保存数据太长"被坐实** | yaml 加 `save_only_model: true`、或 `save_total_limit: 1`、或关 `stage3_gather_16bit_weights_on_model_save` |
| rank 间不对称(有的卡 compute_loss 有的卡 backward) | 真 desync,需要看具体 rank 的栈 | 用 PYTHONFAULTHANDLER 在 SIGABRT 时 dump 的栈对照 |

### 配套现场材料

- `logs/train_<TS>_node<R>.log` —— 主日志,行级时间戳,所有 rank 合并,可 grep `phase=enter`。
- `logs/torchrun_<TS>_node<R>_attempt<N>/<local_rank>/{stdout,stderr}.log` —— 单 rank 原始流,无前缀,适合二分定位。
- `logs/nccl_trace_<TS>_node<R>_rank<L>` —— NCCL flight recorder pickle(watchdog 触发后才生成),用 `python -m torch.distributed.elastic.utils.flight_recorder ...` 解析,能看到 `last_enqueued` / `last_completed` / `seq_id` 三连配对,定位到具体 collective。

## 性能开销

- A 层(env + tee 时间戳):每行 ~0.1ms 的 awk fork 成本,4 节点 32 rank 长跑下 INFO 级日志 ~3 GiB/h(主 log + per-rank log 合计),磁盘开销中等。
- B 层(monkey-patch):每个被 hook 的函数 ~5μs(stderr.write),`forward` 单 step 1 次、`backward` 单 step 1 次、`dataloader_next` 单 step 1 次、`training_step` 单 step 1 次,合计每 step ~30 行 trace,~150μs,**< 0.001%** 单步耗时(单步 60s 量级)。

## 跑稳之后的清理

下次 fabric 稳定能跑完一轮,把以下变量降回:

```bash
NCCL_DEBUG=WARN
TORCH_CPP_LOG_LEVEL=WARNING
TRANSFORMERS_VERBOSITY=warning
TRACE_STAGES=0
```

或直接在脚本里把上面的 `${VAR:-INFO}` 改成 `${VAR:-WARN}` —— B 层的 trace 只在 hang 现场需要,长跑时关掉即可,**不要**直接删 `tools/trace_hook.py` / `tools/trace_train.py`,留作下一次 fabric 抖动时复用。

## 关联文件

| 文件 | 角色 |
|---|---|
| `tools/trace_hook.py` | 阶段级 monkey-patch 模块(本次新增) |
| `tools/trace_train.py` | runpy 包装入口(本次新增) |
| `run_sft_qwen3_5_35b_a3b_base.sh` | 启动脚本(本次改:env / torchrun --tee / awk 时间戳 / entry 切换) |
| `examples/train_full/qwen3_5_35b_a3b_base_pangu_code_data_0408_*.yaml` | 训练 yaml,**未改** |
| `examples/deepspeed/ds_z3_config.json` | deepspeed 配置,**未改** |
| `src/llamafactory/**` | 训练代码,**未改** |
| `docs/qwen3_5_moe_sft_troubleshooting.md` | Bug 索引,Bug 8 系列上下文 |
| `docs/qwen3_5_moe_sft_cluster_vendor_escalation.md` | 升级到供应商的工单模板 |
| `docs/qwen3_5_moe_sft_network_transport_guide.md` | eth0/bond/RoCE 三通路对比 |

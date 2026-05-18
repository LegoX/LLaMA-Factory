# 4 节点训练 fabric 抖动:与集群供应商沟通模板

本文档配套 `qwen3_5_moe_sft_troubleshooting.md` 的 Bug 8 系列(8a~8g)。
当 4 节点训练的 fabric MTBF 退化到无法靠 `save_steps` + auto-retry 自愈时
(经验阈值:MTBF < 30 min,或同 fabric 上单 epoch 都拿不到一次 save),
就该把问题升级到集群供应商。本文档给出工单文本模板、论证逻辑、以及预案
对话(供应商常见反问 + 应答)。

## 何时该升级

不是每次 hang 都升级,优先级评判:

| 现象 | 处置 |
|---|---|
| MTBF > 4h(单 epoch ≥ 1 次 save) | 不升级,靠 auto-retry + save_steps 吸收 |
| MTBF 1-4h | 监控,下调 save_steps;1 周内多次复现再升级 |
| MTBF < 30 min,且对比历史明显退化 | **立即升级**(本文档场景) |
| MTBF < 30 min,但历史就是这样 | 升级时强调"该集群是否 fit 该 workload" |

判 MTBF 退化的硬证据:同样配置(脚本/yaml/NCCL env 全部 git 可追溯),
**前后两个时间窗口 MTBF 差 > 5×**。

本仓库 8b vs 5/17 的对比即此场景:

| 时间 | 配置 | MTBF | 步数/attempt |
|---|---|---|---|
| 2026-05-16 上午 | socket-on-eth0 | **6h44m** | 388 |
| 2026-05-17 全天 | 同上,**0 改动** | **~25 min** | 11-28 |

## 工单文本模板(可直接粘贴)

```
集群:Singapore(sg)region,4 节点 SFT 训练
节点 hostname:e02-sg-cc54llqph1a / cc54llqph15 / cc54llqph12 / cc54llqph14
每节点配置:8×H800 + 8×ConnectX RoCE bond(mlx5_bond_0..7,200Gb/s),
            走 eth0 控制面

## 现象
4 节点 32 卡 PyTorch + DeepSpeed ZeRO-3 训练,跑 ~10-30 分钟后,
所有 32 个 rank 同时卡在 NCCL `_REDUCE_SCATTER_BASE`(NumelIn=526336,~1MiB),
600 秒 watchdog 超时被强制 kill。状态完全对称:每个 rank 的
`last_enqueued = N+2 / last_completed = N-1 / 全员 stuck on collective #N`,
不存在 desync 或单 rank 故障。

集合通信完全卡死的现象在两条数据面都复现:
- 走 RoCE v2(NCCL_IB_HCA=mlx5_bond_0..7, GID_INDEX=3): MTBF ~12 min
- 走纯 TCP socket on eth0(NCCL_IB_DISABLE=1): MTBF 5/16 时 ~4-7h,
  5/17 退化到 ~25 min

## 关键时间点
- 2026-05-16 上午:同样配置 socket-on-eth0,跑了 6h44m / 388 步才挂
- 2026-05-17 全天:同样配置,只能跑 ~25min,挂 12 次以上
→ 5/16 晚到 5/17 之间 fabric 出现退化,**训练侧配置无变更**
  (脚本与 yaml 的 git diff 可提供)

## 已排除的可能
- 单机 8 卡跑同样 workload 24h+ 0 hang(排除节点本机问题)
- 同样 hang 在 RoCE 和 TCP 两条独立通路上都出现(排除某条物理链路或 NIC)
- 32 rank 完全对称卡在同一 SeqNum(排除计算 desync / OOM / 单 rank 故障)
- Loss 在 hang 前持续健康下降(排除模型/数据问题)

## 请协助核查的项
1. 5/16 → 5/17 这段时间,以下 4 节点所在 ToR/Spine 交换机端口
   是否有配置变更、邻居流量异常、链路 flap?
2. 端口计数器(过去 48h):
   - PFC pause frames(TX/RX,按 priority class 拆开)
   - ECN-CE marked packets / WRED drops
   - 物理层 CRC errors / symbol errors / link flap 次数
3. RoCE v2 端到端是否仍按预期配置:
   - DSCP→PCP 映射是否完整
   - PFC priority class 3(RoCE 流量)是否启用且无溢出
   - ECN 是否在 spine 上做了 CC 标记
4. 4 个节点之间的端到端 RTT / packet loss / jitter
   (建议供应商侧主动跑一次 mtr/qperf,我们容器内 ibstat/show_gids 都没装)
5. 集群是否在该时段有其他大流量训练任务(incast 拥塞)?

## 我们这边的诊断材料
- 4 节点同步 hang 日志(每 rank 的 NCCL flight recorder dump,可提供)
- 节点 RoCE 拓扑探测(/sys/class/infiniband/* 全 8 张 ACTIVE/200Gb/sec)
- TORCH_NCCL_DEBUG 完整 log(可提供 INFO 级别下次 hang 现场)
```

## 工单背后的论证逻辑

### 1. 用 NCCL 签名定性,不要泛说"卡了"

供应商最常见的甩锅方向是"训练框架 bug"。`32/32 rank 完美对称卡同一
SeqNum + last_enqueued/last_completed 差固定为 3` 这个签名**只能由底层
网络制造**:

- 计算 desync 时 SeqNum 会错开(某些 rank 跑得快、某些慢)
- 某 rank OOM 时它会先 crash 而不是参与 collective
- 单 rank NIC 故障时只有该 rank 异常,其他 rank 报"unable to recv from peer X"

把这段证据写死,对方就没法推回来。

### 2. "两条独立通路同时挂"是杀手锏

RoCE 走 mlx5_bond_*(verbs/RDMA stack),TCP 走 eth0(kernel socket stack),
两套完全独立的协议栈和驱动。**两边都挂同样形态**,只能是物理 fabric 或
交换机层出问题,不可能是端侧。

这个论证一旦摆上,供应商只能往交换机/线路侧查。

### 3. 单机 0 hang 的对比要写出来

证明节点本身、GPU、NCCL、PyTorch、DeepSpeed、模型代码都健康。供应商一看
"单机稳定 / 多机挂"就只能往跨机方向(交换机/链路)查。

### 4. MTBF 退化曲线 + 配置无变更

最强的钩子。**同样的 git commit、同样的脚本、MTBF 掉 16x**——必然是供应商
侧某处有动作:配置 push、固件升级、邻居业务变化、链路降级。让他们查
change log。

### 5. 点名具体计数器

不要只说"看看网络",要点名 PFC pause / ECN / CRC / link flap 这几个具体
计数器。RoCE 跑不稳的标准信号:

- **PFC pause 飙高** → 接收端拥塞反压,典型 RoCE 死锁前兆
- **ECN-CE 标记多但 CC 没起作用** → 流控配置不全
- **CRC errors / link flap** → 物理层(光模块、线、端口)
- **WRED drops on lossy class** → RoCE 流量被错分到有损队列

### 6. 容器内工具缺失要主动告知

容器里 `ibstat / rdma / show_gids / sminfo / mtr / qperf` 通常都没装,
不要等供应商丢一句"先 ibstat 看看"才发现。一开始就说清楚:数据面信息
我们只能从 `/sys/class/infiniband/*` sysfs 读,需要他们从交换机侧或带
管理网的母机上跑诊断。

## 预案:供应商常见反问 + 应答

| 他们说 | 怎么回 |
|---|---|
| "你这是 NCCL 配置问题" | 同样 NCCL 配置在 5/16 跑了 6h44m / 388 步,没变过。给出 git log 证明 |
| "ZeRO-3 通信量太大" | 32 rank ZeRO-3 跑同 model 在别的集群是稳定的;且 hang 的 collective 只有 1MiB(NumelIn=526336),不是大块 |
| "你单机也跑跑试试" | 已经跑过,稳定 24h+ 0 hang。日志可提供 |
| "重启节点 / 换节点试试" | 4 个节点轮换都试过,挂的形态完全一样,排除单点 |
| "升级 NCCL/OFED 版本" | NCCL 2.27.3 / OFED 最新,且 5/16 同版本是稳的 |
| "训练再跑一次看看" | 已跑 12 次以上,挂的步数都在 270-290 narrow band。统计上不是偶发 |
| "调大 NCCL 超时" | 已经把 sub-PG watchdog 抬到 1800s 实测过(8d),collective 跑满 1800s 也不自愈,纯网络死锁 |

应答原则:每条都给**具体数据 + 时间窗口 + 可提供的日志证据**,不接受
"你再试试"这种甩回式回复。

## 同时该做的事

1. **要 RoCE 拓扑图和 PFC/ECN 配置 dump**——下次想再试 RoCE 时,能先确认
   机房侧的配置确实"做齐了"。RoCE 不是装好就能跑,必须 lossless 端到端配齐
   (DSCP→PCP→queue 全链路、PFC enable、ECN CC、buffer 配置)。
2. **问"同 fabric 其他大客户"近期变化**——共享集群的瞬态死锁经常是被
   某个新上线的高频流量任务拖垮的(incast),供应商有责任做隔离或 QoS。
3. **要 SLA 承诺**——4 节点训练在该 region 的 MTBF 目标是多少,低于
   目标他们补偿什么。

## 工单提交后该提供的材料

供应商一旦响应,他们大概率会要这些数据,先准备好:

```bash
# 1. 4 个 node 的 NCCL flight recorder dump(下次 hang 时自动生成,
#    路径由 TORCH_FR_BUFFER_SIZE 控制,默认在 /tmp 或 cwd)
find / -name "nccl_trace_rank_*" 2>/dev/null

# 2. 4 个 node 各自最近一次 hang 的完整 train log
ls logs/train_<latest_ts>_node{0..3}.log

# 3. RoCE 拓扑探测(本仓库已有的 sysfs 探测脚本输出)
cat sysfs_output.txt
cat ib_detect_output.txt

# 4. 训练侧配置 git diff(证明配置无变更)
git log --all --oneline run_sft_qwen3_5_35b_a3b_base.sh \
        examples/train_full/qwen3_5_35b_a3b_base_*.yaml \
        examples/deepspeed/ds_z3_config.json | tail -20

# 5. 单机 vs 多机 MTBF 对照表
#    (从 docs/qwen3_5_moe_sft_troubleshooting.md 8e/8f/8g 节摘)

# 6. 下次 hang 时主动用 INFO 级 NCCL 抓一次现场
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL \
  bash run_sft_qwen3_5_35b_a3b_base.sh
```

## 一边等供应商一边能做的

升级工单不等于停训,在等响应期间:

- `save_steps` 压到 10(见主 troubleshooting 文档),让 retry 至少能前进
- 不要再调 NCCL_IB_*(8e 已确认无效)
- 不要再 patch DeepSpeed sub-PG timeout(8g 已撤回)
- 不要碰 yaml `ddp_timeout`(对 ZeRO-3 子 PG 无效)
- 如果总进度实在追不上 deadline,临时回 2 节点 + 拉长训练时间
  (8b 阶段 2 节点 RoCE 反而比 4 节点 socket 还快;若供应商一直不响应,
  先用 2 节点保证训练能往前推)

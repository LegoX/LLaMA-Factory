# 多机训练通信通路详解(本机房 H800 集群)

本文档总结本机房(Singapore region,4 节点 H800 SFT 训练环境)上**所有可用的
跨节点通信通路**、各自的物理/协议栈走向、配置方法、以及实测的稳定性差异。
配套主 troubleshooting 文档 Bug 8 系列(8a~8h)和 vendor escalation 模板使用。

读完这篇,你应该能回答:
- 同一台节点上有几种"过网线"的方法?
- 哪些是物理路径、哪些是软件虚拟化路径?
- NCCL 怎么选?怎么强制切?
- 为什么 RoCE 配齐了还会 hang?
- 为什么"socket-on-eth0"和"socket-on-bond"是两件完全不同的事?

---

## 1. 拓扑速览

每个训练节点容器内部的**真实视图**(从 `/sys/class/net` + `/proc/net/bonding` +
`/sys/class/infiniband` 读出):

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Container (e.g. e02-sg-cc54llqph1a)                                     │
│                                                                          │
│  ┌─ 管理面 / 控制面 ────────────────────────────────────────────────┐   │
│  │  eth0  (virtio_net, 200G 标称, MTU 1500)                          │   │
│  │   IP 192.168.0.81  →  default via 192.168.0.253                  │   │
│  │   ↑                                                                │   │
│  │   走 host vhost-user → host kernel TCP/IP → host pNIC             │   │
│  │   K8s API、容器内 DNS、外网、wandb、HF Hub、log push 全在这条上 │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─ 数据面(SR-IOV 直通)───────────────────────────────────────────┐   │
│  │                                                                    │   │
│  │  bond0..bond7 (内核 bonding, mode=802.3ad LACP, 400G, MTU 9000)   │   │
│  │     │                                                              │   │
│  │     ├─ slaves: 每根 bond = 2 × reth (200G + 200G)                 │   │
│  │     │                                                              │   │
│  │     │  bond0  ← reth0 (NUMA 0, PCI 0000:09:00.0)                  │   │
│  │     │         ← reth1 (NUMA 0, PCI 0000:09:00.1)                  │   │
│  │     │  bond1  ← reth2/reth3   (NUMA 0)                            │   │
│  │     │  bond2  ← reth4/reth5   (NUMA 0)                            │   │
│  │     │  bond3  ← reth6/reth7   (NUMA 0)                            │   │
│  │     │  bond4  ← reth8/reth9   (NUMA 1)                            │   │
│  │     │  bond5  ← reth10/reth11 (NUMA 1)                            │   │
│  │     │  bond6  ← reth12/reth13 (NUMA 1)                            │   │
│  │     │  bond7  ← reth14/reth15 (NUMA 1)                            │   │
│  │     │                                                              │   │
│  │     │  IPs (8 个独立 /29 子网,各自 gateway):                    │   │
│  │     │   bond0: 200.21.3.210/29  via 200.21.3.209                  │   │
│  │     │   bond1: 200.21.4.138/29  via 200.21.4.137                  │   │
│  │     │   bond2: 200.21.5.90/29   via 200.21.5.89                   │   │
│  │     │   bond3: 200.21.6.54/29   via 200.21.6.53                   │   │
│  │     │   bond4: 200.21.7.22/29   via 200.21.7.21                   │   │
│  │     │   bond5: 200.21.7.242/29  via 200.21.7.241                  │   │
│  │     │   bond6: 200.21.8.202/29  via 200.21.8.201                  │   │
│  │     │   bond7: 200.21.9.146/29  via 200.21.9.145                  │   │
│  │     │   每根 bond → 不同 ToR/leaf → spine 8 路 ECMP               │   │
│  │     ↓                                                              │   │
│  │  reth0..reth15 (mlx5_core, 16 张物理 ConnectX-7 口, 200G, MTU 9k) │   │
│  │     ↓                                                              │   │
│  │  mlx5_bond_0..7 (verbs/RDMA 抽象,跟 bond0..7 物理同设备)         │   │
│  │     port=1 link=Ethernet ACTIVE, rate=200 Gb/sec (2X NDR)          │   │
│  │     RoCE v2, GID v2 idx=1, fw=32.39.3920, hca=MT41692              │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**一台节点上同一组物理 NIC 在 kernel 里有 3 套不同抽象**:
- `reth0..reth15` —— mlx5 物理 netdev,16 个独立端口
- `bond0..bond7` —— 8 根 LACP bond,把 reth 两两聚合,kernel TCP 视角
- `mlx5_bond_0..7` —— OFED RDMA 视角,verbs API 用的设备名

**3 个名字指向同一组硬件,但走不同协议栈**——这是后面 3 条通路区别的根源。

---

## 2. 三条通路详解

### 通路 A:socket on eth0(virtio,默认)

**协议栈**:

```
GPU → host RAM → vhost-user ring → [host kernel TCP/IP] → [host pNIC] → 物理网络
                  └─ 软件虚拟化层,所有租户/邻居共享                     ─┘
```

**特性**:
- **唯一不直通物理网卡**的通路,数据包在 host kernel 走一遍
- MTU 1500,标称 200G(实际是 host pNIC 共享带宽,不是独享)
- **K8s/管理面/外网/DNS 都默认走它**(看 `/proc/net/route` 第一条 default via 192.168.0.253)
- 任何同 host 邻居容器/管理面流量打满 → eth0 socket 跟着卡
- 历史最稳但**易受 host 邻居影响**:5/16 撑了 6h44m / 388 步,5/17 同配置 0 改动退化到 ~25min(**怀疑邻居容器开始上量**)

**何时用**:
- 单/双节点训练且 fabric 配置未知 → 默认它最不容易踩坑
- 集群刚上线、还没确认 RoCE/bond 配置完整时
- 故障兜底

**配置**(脚本 `NCCL_TRANSPORT=eth0`,默认值):
```bash
NCCL_SOCKET_IFNAME=eth0
GLOO_SOCKET_IFNAME=eth0
TP_SOCKET_IFNAME=eth0
# 不设 NCCL_IB_DISABLE — NCCL 会自动 detect 没有可用 IB 后 fallback 到 socket
```

---

### 通路 B:socket on bond(mlx5 物理 + kernel TCP)

**协议栈**:

```
GPU → 容器 kernel TCP/IP → [SR-IOV 直通 mlx5_core] → [bond LACP] → ToR/leaf → spine
                                                      └─ 8 根并行 ─┘
```

**特性**:
- **物理路径**:SR-IOV 把 mlx5 NIC 直通给容器,**不经 host kernel**
- 8×400G LACP 多 rail,每根 bond 接独立 ToR(spine 侧 8 路 ECMP)
- 走 kernel TCP 栈(不是 verbs),**TCP CUBIC 自带拥塞控制 + ECN**,无 PFC 死锁风险
- LACP 哈希策略 = `layer3+4`(src/dst IP + L4 port),**单 TCP 流只走一个 200G slave**;
  → 必须靠 NCCL 多 socket(`NCCL_NSOCKS_PERTHREAD × NCCL_SOCKET_NTHREADS`)
  把 src port 多样化,LACP 才能把流量摊到 2 个 slave 上
- MTU 9000(jumbo),减少 syscall 开销
- **NUMA 对齐**:NUMA 0 GPU(0000:08/7e/a2/c6) 用 bond0..3, NUMA 1 GPU(0001:09/7f/a3/c7) 用 bond4..7
- kernel **默认 metric** 1010..1017 递增,普通进程往 200.21.0.0/16 发只走 bond0;
  → NCCL 用 `SO_BINDTODEVICE` 显式绑每个 socket 到指定 bond,**绕过 kernel 默认 routing**,
    所以 `NCCL_SOCKET_IFNAME` 必须**列全 8 根**才能真正多 rail 并行

**何时用**:
- eth0 退化但 RoCE 配置不可信时(本机房现状,8h 引入)
- 想绕开 host 共享路径但又不想踩 PFC/ECN 死锁

**配置**(脚本 `NCCL_TRANSPORT=bond`):
```bash
NCCL_IB_DISABLE=1                                                    # 关 IB,强制 socket
NCCL_SOCKET_IFNAME=bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7   # 必须全列
GLOO_SOCKET_IFNAME=bond0                                              # rdzv 单根够用
NCCL_SOCKET_NTHREADS=8                                                # 8 线程 × 8 sock = 64 路
NCCL_NSOCKS_PERTHREAD=8                                               # = LACP 哈希足够多样
```

**何时**不该**用**:
- 8 根 bond 没全部配 IP 时(NCCL 会 fallback 到单 rail,等于退化为 400G 单流)
- 跨节点 bond 子网不可路由时(很少见,但要先 ping 测)

---

### 通路 C:RDMA on mlx5_bond_*(RoCE v2 verbs)

**协议栈**:

```
GPU 显存 → [nvidia_peermem] → [mlx5 verbs] → [SR-IOV mlx5] → ToR (无损 RoCE) → spine
└─ GPUDirect: 不走 host RAM, 不走 host CPU ─┘                  └─ 需端到端 PFC + ECN ─┘
```

**特性**:
- **延迟最低 / 带宽最高的理论通路**:GPU 显存直接 DMA 到 NIC,bypass host
- 端到端走 RDMA over Converged Ethernet v2(RoCE v2),**走 verbs**而不是 socket
- **依赖无损以太网(lossless Ethernet)**:DSCP→PCP→queue 全链路、PFC、ECN CC 必须配齐
- PFC 死锁风险:接收方反压(pause frames)倒灌 → spine 拥塞 → 整张 fabric 瘫痪
- **本机房实测 hang ~12min**(8c/8d run),很可能是 PFC/ECN 配置不全 + 之前 GID idx 选错(idx=3 在本机不是 v2 全局 GID,本机正确值是 idx=1)

**关键配置项**:

| 变量 | 含义 | 本机房正确值 |
|---|---|---|
| `NCCL_IB_HCA` | 用哪些 HCA | `mlx5_bond_0,...,mlx5_bond_7` |
| `NCCL_IB_GID_INDEX` | GID 表中第几项 | **3**(`/sys/class/infiniband/mlx5_bond_0/ports/1/gids/3` 内容是 `::ffff:c815:03d2`,即 IPv4-mapped 形式的 200.21.3.210,与 bond0 IPv4 一致;**type=RoCE v2 + IPv4-mapped**,跨 /29 spine 可路由。**注意**:idx=1 也是 type=RoCE v2,但 gid 是 link-local IPv6 `fe80::...`,跨子网不可路由——选错就 `ibv_modify_qp -> ENETUNREACH(101)`) |
| `NCCL_IB_TC` | DSCP 流量类 | 160(对应 PCP 5 / RoCE 队列) |
| `NCCL_IB_SL` | Service Level | 3(RoCE priority class) |
| `NCCL_SOCKET_IFNAME` | rdzv/启动期 socket | `eth0`(只用一根 rdzv) |

**何时用**:
- PFC + ECN 经过供应商工单确认全配齐之后
- 大模型训练且单步耗时已被通信主导(算/通比 < 1)
- A/B 对照(永远值得跟 socket 对比一次)

**何时不该用**:
- 没拿到供应商 PFC counter / ECN 配置 dump 之前 → 别压在 production run 上
- 仅有 1-2 节点训练 → 通信量小,RoCE 优势不明显,不值得冒死锁风险

---

## 3. 实测对比表

| 维度 | A. eth0 | B. bond(8h 新) | C. RoCE |
|---|---|---|---|
| 物理路径 | virtio 软栈(host 共享) | mlx5 直通(独占) | mlx5 直通(独占) |
| 协议栈 | host kernel TCP | container kernel TCP | OFED verbs / RDMA |
| 流控 | TCP CUBIC | TCP CUBIC | PFC + ECN(可死锁) |
| MTU | 1500 | 9000 jumbo | 9000 jumbo |
| 单流上限 | ~200G(共享) | 200G(LACP slave) | 200G(单 HCA) |
| 8 流并发 | ~200G(共享) | **~3.2T**(8×400) | ~1.6T(8×200) |
| GPUDirect | ✗(过 host RAM) | ✗(过容器 kernel) | ✓(直挂显存) |
| NUMA 亲和 | ✗ | ✓ | ✓ |
| 配置依赖 | 极低 | 8 根 bond IP 齐 | PFC + ECN 全链路 |
| 死锁风险 | 无 | 无 | **高** |
| 本机房 MTBF | 5/16=6h44m, 5/17=25min | 待测(8h) | ~12min(8c/8d) |
| 默认推荐 | 单/双节点 | **4 节点(本机房当前最优)** | 验证 PFC 之后再压 |

---

## 4. 通路诊断:`fabric_test.sh`

容器内**没有 `ip` / `ifconfig` / `ibstat` / `mlnx_qos`** 等标准网络工具。
所有信息只能从 `/sys` 和 `/proc` 读。已落地的探测脚本是
仓库根目录的 `fabric_test.sh`,在训练节点跑一次能拿到:

| 节 | 取自 | 关键判断 |
|---|---|---|
| [1] netdev 概览 | `/sys/class/net/*/{driver,operstate,carrier,speed,mtu,master,address,device/numa_node}` | 区分 virtio_net vs mlx5_core,看 LACP slave 关系 |
| [2] bond 拓扑 | `/proc/net/bonding/*` | LACP mode、hash policy、slave up/down、链路失败计数 |
| [3] IB → netdev 映射 | `/sys/class/infiniband/*/{device/net,ports/*/state,rate,link_layer,gid_attrs/types/*}` | GID v2 索引、port ACTIVE 状态、HCA 型号 |
| [4]+[5] IP / 路由 | `/proc/net/{fib_trie,route}` | bond IP 分配,默认 gateway,multi-rail 子网设计 |
| [6] PFC / ECN | `/sys/class/net/<dev>/qos/{pfc_enable,trust_state,dscp_app_index}` | 新版 OFED 不暴露,**必须从供应商交换机侧** dump |
| [7] GPU↔NIC NUMA | `/sys/class/drm/card*/device/{numa_node,driver}` + NIC 同前 | 决定能否走 GPUDirect RDMA peer 跨 socket |
| [8] NCCL env 当前值 | `env \| grep NCCL_/GLOO_/TP_/UCX_` | 排除遗留 env 污染 |
| [9] kernel 模块 | `/sys/module/{mlx5_core,mlx5_ib,ib_core,nvidia_peermem}/version` | nvidia_peermem 必须加载否则 GPUDirect 不可用 |

### 跑法

```bash
bash /jyx_data/LLaMA-Factory-latest/fabric_test.sh
```

### 容易看错的输出

1. **`bond0..bond7` 的 `drv=?`**:bonding 驱动不暴露 `/device/driver` 链接,这是正常的,不代表 bond 没起来——看 `state=up car=1 sp=400000` 才是真状态
2. **`mlx5_bond_X netdev=rethY`**:只列出一个 slave 名字,实际 bond 两根 reth 都参与 verbs 流量,不是 sysfs 偷工减料
3. **fib_trie 一坨 `172.16.x.x`(>150 个)**:K8s ClusterIP 被 kube-proxy 注入,**不是真本机 IP**;真本机 bond IP 是 `200.21.{3,4,5,6,7,7,8,9}.x` 这 8 个
4. **`mlx5_ib / ib_core / ib_uverbs / rdma_cm` 显示 `<not loaded>`**:`/sys/module/<x>/version` 文件缺失不等于没加载;`/sys/class/infiniband/mlx5_bond_*` 8 个设备能枚举出来,就证明 RDMA 栈是好的(模块编译进 in-tree 或 strip 过 modinfo)
5. **`pfc_enable=?`**:新版 mlx5 OFED 把 QoS 接口移到了 `mlnx_qos` 工具,sysfs 不再暴露——容器内**永远查不到 PFC 真实状态**,只能从交换机侧

---

## 5. NCCL 怎么 pick 通路

NCCL 启动时会扫所有可用 transport,优先级大致:

```
NVLink (节点内 GPU↔GPU, P2P)  >  IB/RDMA (verbs)  >  Socket (kernel TCP)
```

**控制变量**:

| 变量 | 作用 |
|---|---|
| `NCCL_IB_DISABLE=1` | 关 IB,强制走 socket |
| `NCCL_IB_HCA` | 指定用哪些 HCA(verbs) |
| `NCCL_IB_GID_INDEX` | RoCE v2 全局 GID 索引(本机房 = 3) |
| `NCCL_SOCKET_IFNAME` | socket 模式用哪些 netdev,逗号分隔可多 rail |
| `NCCL_NSOCKS_PERTHREAD` × `NCCL_SOCKET_NTHREADS` | socket 模式并发流数 |
| `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,NET` | 看到底用了哪条通路 |

**怎么验证 NCCL 真的用了想用的通路**:

```bash
# 临时打开 NCCL INFO 跑一次 attempt,grep 这两行
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET ... bash run_sft_qwen3_5_35b_a3b_base.sh 2>&1 \
  | grep -E "NET/Socket : Using|NET/IB : Using"
```

期望输出(bond 模式):

```
NET/Socket : Using [0]bond0:200.21.3.210<0> [1]bond1:200.21.4.138<0> ... [7]bond7:200.21.9.146<0>
```

8 行 = multi-rail 真的并行起来。如果只有 `[0]bond0`,说明 `NCCL_SOCKET_IFNAME`
没传过去(env scope / launcher 吃掉了),需要查 launcher。

期望输出(roce 模式):

```
NET/IB : Using [0]mlx5_bond_0:1/RoCE [1]mlx5_bond_1:1/RoCE ... [7]mlx5_bond_7:1/RoCE
```

---

## 6. 关键陷阱与教训

### 6.1 `set -euo pipefail` 杀诊断函数

`dump_net_topo()` 里 `readlink 2>/dev/null | xargs -r basename` 这种结构,
当 `readlink` 失败(比如 bond 没有 `device` 子目录),`pipefail` 让整条 pipe
返回非零,叠加 `set -e` 主脚本立即退出。

**症状**:torchrun 还没起,脚本就静默退到 prompt,`[net-topo] === ...` 标题之后全空。

**修复**:诊断函数入口主动 `set +e; set +o pipefail`,出口前恢复并 `return 0`。
诊断本来就该是 best-effort 语义,不该阻断主流程。

### 6.2 LACP layer3+4 哈希 → 单 socket 不分担

bond 的哈希策略默认是 `layer3+4`,**对单条 TCP 连接,哈希结果固定**,
整条流只走一个 200G slave。要靠 NCCL 多 socket(不同 src port 让哈希结果散开)
才能把 2 个 slave 都喂满。

**所以 `NCCL_NSOCKS_PERTHREAD=8 × NCCL_SOCKET_NTHREADS=8` 必须设**,
不开就只能用一半带宽。

### 6.3 GID 索引必须读 GID 内容,不能只读 type

`NCCL_IB_GID_INDEX=3` 是某些集群的历史默认,**不通用**;**但只读 sysfs 的 type 字段
也不够**。同一张 HCA 的 sysfs 通常会同时暴露多条 type=`RoCE v2` 的 GID:

| 形态 | gid 内容 | 是否可跨 spine 路由 |
|---|---|---|
| link-local IPv6 | `fe80:0000:...` (基于 MAC,EUI-64) | ✗ |
| IPv4-mapped IPv6 | `0000:...:ffff:c815:....` (后 4 段 hex 是 bond IPv4) | ✓ |
| global IPv6 | `2xxx:...`(若集群分配了) | ✓ |

本机房:
- idx=0 = IB/RoCE v1 + link-local(过时,不用)
- idx=1 = RoCE v2 + link-local(**踩坑点**:type 对了但 gid 跨子网不可达,
  跑出 `ibv_modify_qp -> ENETUNREACH(101)` 直接退)
- idx=2 = IB/RoCE v1 + IPv4-mapped(过时,不用)
- **idx=3 = RoCE v2 + IPv4-mapped → 唯一可用**

**所以读法是双字段联合**:
```bash
for i in $(ls /sys/class/infiniband/mlx5_bond_0/ports/1/gids); do
  g=$(cat /sys/class/infiniband/mlx5_bond_0/ports/1/gids/$i)
  t=$(cat /sys/class/infiniband/mlx5_bond_0/ports/1/gid_attrs/types/$i)
  printf "idx=%s type=%s gid=%s\n" "$i" "$t" "$g"
done | grep "RoCE v2" | grep -v "fe80:"
# 留下来那条的 idx 才是要传给 NCCL_IB_GID_INDEX 的值
```

### 6.4 默认 gateway 在 eth0 → 任何"没绑接口"的流量都走 virtio

```
/proc/net/route 第一条:  default  via 192.168.0.253  dev eth0
```

意味着:外网下载、wandb、HF Hub、容器内 DNS、所有 host 共享流量都走 eth0。
即使 NCCL 切到 bond,**rdzv master 的 hostname 解析、log push、wandb sync 仍走 eth0**。
这本身没问题(rdzv 流量极小),但**如果 eth0 烂了,rdzv 也起不来**,
所以"切到 bond 模式 = 数据面完全脱离 eth0"是不准确的——只有 NCCL 集合通信脱离了。

### 6.5 NCCL_SOCKET_IFNAME 必须列全

NCCL 不会自动发现 8 根 bond,只用 `NCCL_SOCKET_IFNAME` 里**显式列出**的接口。
列错或少列 → silently fallback 到 kernel default routing → 全压到 metric 最小的
bond0 → 只有 1/8 带宽。

**列全的语法**(逗号分隔,无空格):
```
NCCL_SOCKET_IFNAME=bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7
```

### 6.6 eth0 ≠ 物理网卡

容器里看到的 `eth0` 是 `virtio_net` 驱动,本质是宿主机 vhost-user 透出来的虚拟接口。
"eth0 速率 200G" 是 host 一侧 pNIC 的标称值,**容器实际能用多少看 host 调度**。
任何同 host 邻居容器 / 系统态流量打满,eth0 就跟着退化——这就是为什么
5/16 同配置稳 6h44m 但 5/17 退化到 25min(host 上邻居业务变了)。

### 6.7 PFC 状态容器里查不到

`/sys/class/net/<dev>/qos/pfc_enable` 在 mlx5_core 23.10+ **不再暴露**。
容器内没 `mlnx_qos`、没 `ibstat`、没 `rdma`——查 PFC 唯一办法:
**让供应商从交换机侧 dump counter**(对应 vendor escalation 工单第 2 项)。

### 6.8 NCCL_DEBUG=WARN 看不到通路选择

WARN 级别只在 fallback 失败时输出。要看 NCCL 实际选了哪条通路、绑了哪些接口,
**必须临时开 INFO**(只跑一个 attempt 即可,不要长期开,日志爆量)。

---

## 7. 切换通路速查

```bash
# 默认(eth0,virtio)
NNODES=4 NODE_RANK=$R bash run_sft_qwen3_5_35b_a3b_base.sh

# 切到 bond(物理 socket multi-rail,本机房当前推荐)
NCCL_TRANSPORT=bond NNODES=4 NODE_RANK=$R bash run_sft_qwen3_5_35b_a3b_base.sh

# 切到 RoCE(verbs/RDMA,本机房当前不推荐,等供应商确认 PFC 之后再试)
NCCL_TRANSPORT=roce NNODES=4 NODE_RANK=$R bash run_sft_qwen3_5_35b_a3b_base.sh

# 验证 NCCL 真的用了想用的通路(只跑一个 attempt 拿证据)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_TRANSPORT=bond NNODES=4 NODE_RANK=$R \
  bash run_sft_qwen3_5_35b_a3b_base.sh 2>&1 | grep -E "NET/Socket|NET/IB" | head
```

---

## 8. 何时升级到 vendor

参考 `qwen3_5_moe_sft_cluster_vendor_escalation.md`。简短规则:

- **三条通路都试过,MTBF < 30min** → 立即升级,且要求供应商 dump PFC/ECN/CRC counter
- **bond 模式稳但 RoCE 仍 hang** → 不升级训练,但开工单要求确认 PFC 配置
- **bond 也 ~25min hang** → 不是 virtio 问题,是物理 fabric/spine 真的烂,升级最高优先级

vendor 工单里**两条独立通路同时挂**这个证据是关键(socket-on-bond 和 RoCE-verbs
是完全独立的协议栈,如果两边都挂同样形态,只可能是物理 fabric 层问题)。

---

## 附录:本文档对应代码位置

| 内容 | 文件 |
|---|---|
| 三种 transport 切换实现 | `run_sft_qwen3_5_35b_a3b_base.sh` lines 61-106 |
| 启动期 dump_net_topo | `run_sft_qwen3_5_35b_a3b_base.sh` lines 488-540 |
| 离线诊断脚本 | `fabric_test.sh`(仓库根目录) |
| Bug 8h 节(本文档简版) | `docs/qwen3_5_moe_sft_troubleshooting.md` 8h 节 |
| 升级供应商模板 | `docs/qwen3_5_moe_sft_cluster_vendor_escalation.md` |

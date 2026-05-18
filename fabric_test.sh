{
  echo "=========================================================="
  echo "[1] 物理 netdev 概览(按 driver 分组,过滤虚拟接口)"
  echo "=========================================================="
  for d in /sys/class/net/*; do
      n=$(basename "$d")
      case "$n" in lo|cali*|veth*|docker*|kube-*|nodelocaldns|bonding_masters|tunl*|ip6*|sit*) continue ;; esac
      drv=$(readlink "$d/device/driver" 2>/dev/null | xargs -r basename)
      st=$(cat "$d/operstate" 2>/dev/null)
      car=$(cat "$d/carrier" 2>/dev/null)
      sp=$(cat "$d/speed" 2>/dev/null)
      mtu=$(cat "$d/mtu" 2>/dev/null)
      ms=$(readlink "$d/master" 2>/dev/null | xargs -r basename)
      mac=$(cat "$d/address" 2>/dev/null)
      pci=$(readlink "$d/device" 2>/dev/null | xargs -r basename)
      numa=$(cat "$d/device/numa_node" 2>/dev/null)
      printf "%-12s drv=%-12s st=%-5s car=%s sp=%-7s mtu=%-5s master=%-8s numa=%-3s pci=%-13s mac=%s\n" "$n" "${drv:-?}" "${st:-?}" "${car:-?}" "${sp:-?}" "${mtu:-?}" "${ms:-}" "${numa:-?}" "${pci:-?}" "${mac:-?}"
  done

  echo
  echo "=========================================================="
  echo "[2] bond 拓扑(每个 bond 的 mode + slave 列表)"
  echo "=========================================================="
  for b in /proc/net/bonding/*; do
      [ -f "$b" ] || continue
      echo "--- $(basename "$b") ---"
      grep -E "Bonding Mode|Transmit Hash|MII Status|Slave Interface|Speed|Link Failure Count|Permanent HW 
  addr|Aggregator ID|Number of ports" "$b"
  done

  echo
  echo "=========================================================="
  echo "[3] IB device → netdev 映射 + port state/rate + GID v2"
  echo "=========================================================="
  for ib in /sys/class/infiniband/*/; do
      [ -d "$ib" ] || continue
      ibname=$(basename "$ib")
      nd=$(ls "$ib/device/net" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
      fw=$(cat "$ib/fw_ver" 2>/dev/null)
      hca=$(cat "$ib/hca_type" 2>/dev/null)
      for p in $(ls "$ib/ports" 2>/dev/null); do
          pst=$(cat "$ib/ports/$p/state" 2>/dev/null)
          prate=$(cat "$ib/ports/$p/rate" 2>/dev/null)
          plink=$(cat "$ib/ports/$p/link_layer" 2>/dev/null)
          # 找一个 RoCE v2 的 GID(type 文件值为 'RoCE v2' 或 'IB/RoCE v1')
          v2_idx=""
          for gd in "$ib/ports/$p/gid_attrs/types"/*; do
              [ -f "$gd" ] || continue
              t=$(cat "$gd" 2>/dev/null)
              if echo "$t" | grep -qi "v2"; then
                  v2_idx=$(basename "$gd")
                  break
              fi
          done
          printf "%-15s port=%s link=%-9s st=%-15s rate=%-10s netdev=%-30s gid_v2_idx=%s\n" \
                 "$ibname" "$p" "${plink:-?}" "${pst:-?}" "${prate:-?}" "${nd:-<none>}" "${v2_idx:-?}"
      done
      echo "  fw=$fw hca=$hca"
  done

  echo
  echo "=========================================================="
  echo "[4] 各 bond 的 IPv4(/proc/net/fib_trie 反查)"
  echo "=========================================================="
  awk '
      /^[[:space:]]*\|--/ { ip=$2; next }
      /\/32 host LOCAL/ { print ip }
  ' /proc/net/fib_trie 2>/dev/null | sort -u

  echo
  echo "=========================================================="
  echo "[5] 路由表(/proc/net/route, hex→dec)"
  echo "=========================================================="
  printf "%-10s %-18s %-18s %-6s %-8s %-6s\n" "Iface" "Dest" "Gateway" "Flags" "Metric" "MTU"
  awk 'NR>1 {
      d=$2; g=$3
      printf "%s %s%s%s%s %s%s%s%s %s %s %s\n", $1,
          sprintf("%d.",  ("0x" substr(d,7,2))+0),
          sprintf("%d.",  ("0x" substr(d,5,2))+0),
          sprintf("%d.",  ("0x" substr(d,3,2))+0),
          sprintf("%d",   ("0x" substr(d,1,2))+0),
          sprintf("%d.",  ("0x" substr(g,7,2))+0),
          sprintf("%d.",  ("0x" substr(g,5,2))+0),
          sprintf("%d.",  ("0x" substr(g,3,2))+0),
          sprintf("%d",   ("0x" substr(g,1,2))+0),
          $4, $7, $9
  }' /proc/net/route | awk '{ printf "%-10s %-18s %-18s %-6s %-8s %-6s\n",$1,$2,$3,$4,$5,$6 }'

  echo
  echo "=========================================================="
  echo "[5b] 各 bond/eth 上绑定的 IPv4(直接读 /proc/net/fib_trie 上下文)"
  echo "=========================================================="
  # 把每个 LOCAL /32 IP 与它所属 device 关联(通过 fib_trie 的 dev 字段)
  awk '
      /^Local:/ { in_local=1; next }
      /^Main:/  { in_local=0 }
      in_local && /^[[:space:]]*\|--/ { ip=$2 }
      in_local && /\/32 host LOCAL/ { print ip }
  ' /proc/net/fib_trie | sort -u
  echo "--- (dev 归属反查 via /proc/net/arp 不够,但每根 bond 的 IP 一般和 reth slave 一一对应,见 [4]) ---"

  echo
  echo "=========================================================="
  echo "[6] PFC / ECN 配置(mlx5 only,通过 sysfs)"
  echo "=========================================================="
  for d in /sys/class/net/*; do
      n=$(basename "$d")
      drv=$(readlink "$d/device/driver" 2>/dev/null | xargs -r basename)
      [ "$drv" = "mlx5_core" ] || continue
      pfc_en=$(cat "$d/qos/pfc_enable" 2>/dev/null)
      trust=$(cat "$d/qos/trust_state" 2>/dev/null)
      dscp_app=$(cat "$d/qos/dscp_app_index" 2>/dev/null | tr '\n' ' ')
      printf "%-12s pfc_enable=%-12s trust=%-6s dscp_app=%s\n" \
             "$n" "${pfc_en:-?}" "${trust:-?}" "${dscp_app:-?}"
  done

  echo
  echo "=========================================================="
  echo "[7] GPU ↔ NIC NUMA 亲和(同 numa_node 才能走 GPUDirect RDMA)"
  echo "=========================================================="
  for g in /sys/class/drm/card*/device; do
      [ -f "$g/numa_node" ] || continue
      pci=$(readlink "$g" 2>/dev/null | xargs -r basename)
      nn=$(cat "$g/numa_node" 2>/dev/null)
      drv=$(readlink "$g/driver" 2>/dev/null | xargs -r basename)
      [ "$drv" = "nvidia" ] || continue
      printf "GPU pci=%-13s numa=%s\n" "$pci" "$nn"
  done
  echo "---"
  for d in /sys/class/net/*; do
      n=$(basename "$d")
      drv=$(readlink "$d/device/driver" 2>/dev/null | xargs -r basename)
      [ "$drv" = "mlx5_core" ] || continue
      pci=$(readlink "$d/device" 2>/dev/null | xargs -r basename)
      nn=$(cat "$d/device/numa_node" 2>/dev/null)
      printf "NIC %-8s pci=%-13s numa=%s\n" "$n" "$pci" "$nn"
  done

  echo
  echo "=========================================================="
  echo "[8] NCCL 相关 env(当前 shell 可见值)"
  echo "=========================================================="
  env | grep -E '^(NCCL_|GLOO_|TP_|UCX_|RDMAV_|MELLANOX_|NVIDIA_|CUDA_VISIBLE)' | sort

  echo
  echo "=========================================================="
  echo "[9] 内核 fabric 相关模块 + 版本"
  echo "=========================================================="
  for m in mlx5_core mlx5_ib ib_core ib_uverbs rdma_cm nvidia_peermem nv_peer_mem; do
      f=/sys/module/$m/version
      [ -f "$f" ] && printf "%-15s %s\n" "$m" "$(cat $f)" || printf "%-15s <not loaded>\n" "$m"
  done

  echo
  echo "=========================================================="
  echo "[10] OFED / NCCL / 内核摘要"
  echo "=========================================================="
  uname -r
  [ -f /etc/mlnx-release ] && cat /etc/mlnx-release
  [ -f /opt/mellanox/VERSION ] && cat /opt/mellanox/VERSION
  python -c "import torch;print('torch=',torch.__version__,'nccl=',torch.cuda.nccl.version())" 2>/dev/null
  } 2>&1
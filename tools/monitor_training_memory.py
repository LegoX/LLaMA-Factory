# Copyright 2025 the LlamaFactory team.
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

"""Bounded, read-only host/cgroup/GPU memory sampling; no torch import."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def read_fields(path):
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.replace(":", " ").split()
            if len(parts) >= 2 and parts[1].isdigit():
                result[parts[0]] = int(parts[1])
    except OSError:
        pass
    return result


def cgroup_memory():
    # Resolve the cgroup mount root, including containers with a private namespace.
    groups = Path("/proc/self/cgroup").read_text().splitlines()
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    for group in groups:
        _, controllers, relative = group.split(":", 2)
        kind = "cgroup2" if not controllers else "cgroup"
        if controllers and "memory" not in controllers.split(","):
            continue
        for mount in mounts:
            left, right = mount.split(" - ", 1)
            fields, extra = left.split(), right.split()
            if extra[0] != kind or (controllers and "memory" not in extra[2].split(",")):
                continue
            root, target = fields[3:5]
            suffix = relative[len(root) :] if relative.startswith(root) else relative
            base = Path(target) / suffix.lstrip("/")
            if not base.is_dir():
                continue
            result = {"path": str(base)}
            names = (
                ("memory.current", "memory.max", "memory.peak", "memory.swap.current")
                if kind == "cgroup2"
                else ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.max_usage_in_bytes", "memory.failcnt")
            )
            for name in names:
                try:
                    raw = (base / name).read_text().strip()
                    result[name] = int(raw) if raw.isdigit() else raw
                except OSError:
                    pass
            for name in ("memory.events", "memory.stat"):
                result[name] = read_fields(base / name)
            return result
    return {}


def sample(parent_pid):
    processes = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            status = read_fields(path / "status")
            if "PPid" in status:
                processes[int(path.name)] = status
    descendants = {parent_pid}
    while True:
        added = {pid for pid, status in processes.items() if status["PPid"] in descendants} - descendants
        if not added:
            break
        descendants.update(added)
    rows = []
    for pid in sorted(descendants - {os.getpid()}):
        status = processes.get(pid, {})
        rows.append(
            {"pid": pid, **{k: status.get(k, 0) for k in ("VmRSS", "RssAnon", "RssFile", "VmLck", "VmPin", "VmSwap")}}
        )
    host = read_fields("/proc/meminfo")
    result = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_kib": {
            k: host.get(k)
            for k in ("MemTotal", "MemAvailable", "Cached", "Shmem", "Unevictable", "Mlocked", "SwapFree")
        },
        "processes_kib": rows,
        # RSS includes shared pages repeatedly; use cgroup.current for total usage.
        "rss_sum_kib_including_shared": sum(row["VmRSS"] for row in rows),
        "cgroup_bytes": cgroup_memory(),
    }
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        result["gpu_memory_mib"] = gpu.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_memory_mib"] = []
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    while Path(f"/proc/{args.parent_pid}").exists():
        try:
            print(json.dumps(sample(args.parent_pid)), flush=True)
        except Exception as exc:
            print(json.dumps({"monitor_error": repr(exc)}), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

"""Observable, opt-in host-cache cleanup at completed microbatch boundaries."""
import ctypes
import gc
import json
import os
import time
from pathlib import Path
import torch


def audit_host_memory(output_dir, rank, micro_step):
    release = os.getenv('QWEN38_HOST_CACHE_RELEASE') == '1'
    path = Path(output_dir) / f'host-memory-rank{rank}-micro{micro_step}.json'
    records = []
    def record(phase):
        status = Path('/proc/self/status').read_text().splitlines()
        memory = {line.split(':')[0]:line.split(':')[1].strip() for line in status
                  if line.startswith(('VmRSS:', 'VmHWM:', 'RssAnon:', 'VmLck:'))}
        stats = dict(torch.npu.host_memory_stats()) if hasattr(torch.npu, 'host_memory_stats') else {}
        records.append({'time':time.time(), 'phase':phase, 'process_memory':memory,
                        'pinned_allocator':stats})
    torch.npu.synchronize()
    record('before')
    if release:
        gc.collect()
        torch.npu.synchronize()
        record('after_gc')
        torch.npu.host_empty_cache()
        ctypes.CDLL(None).malloc_trim(0)
        record('after_unused_cache_release')
    path.write_text(json.dumps({'release_enabled':release,'records':records},indent=2))

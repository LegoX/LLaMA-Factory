"""Opt-in CPU storage for PyTorch 2.7 checkpoint recomputation caches.

Keep metadata checking and weak-key lifetimes; change only where cached tensors live.
"""

import contextvars
import functools
import os
import weakref
from collections import defaultdict
from types import SimpleNamespace

import torch


_metadata = contextvars.ContextVar("checkpoint_metadata_only", default=False)
STATS = {"stores": 0, "loads": 0}


class CPUCache(weakref.WeakKeyDictionary):
    def __setitem__(self, key, value):
        if value.device.type == "npu" and value.numel() * value.element_size() >= int(
            os.getenv("PROBE_CHECKPOINT_CACHE_MIN_BYTES", str(2**20))
        ):
            STATS["stores"] += 1
            value = SimpleNamespace(
                device=value.device, shape=value.shape, dtype=value.dtype, cpu=value.detach().to("cpu")
            )
        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, SimpleNamespace):
            if _metadata.get():
                return value
            STATS["loads"] += 1
            return value.cpu.to(value.device)
        return value


def install_checkpoint_cpu_cache():
    import torch.utils.checkpoint as cp

    if getattr(cp._CheckpointFrame, "_cpu_cache_installed", False):
        return
    if not torch.__version__.startswith("2.7.1"):
        raise RuntimeError("Checkpoint CPU cache validated only against installed torch 2.7.1")
    original_init = cp._CheckpointFrame.__init__
    original_check = cp._CheckpointFrame.check_recomputed_tensors_match

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.metadata_fn not in [None, cp._default_meta_extractor]:
            raise ValueError("Unsupported custom checkpoint metadata function")
        self.recomputed = defaultdict(CPUCache)

    @functools.wraps(original_check)
    def check(self, *args, **kwargs):
        token = _metadata.set(True)
        try:
            return original_check(self, *args, **kwargs)
        finally:
            _metadata.reset(token)

    cp._CheckpointFrame.__init__ = init
    cp._CheckpointFrame.check_recomputed_tensors_match = check
    cp._CheckpointFrame._cpu_cache_installed = True

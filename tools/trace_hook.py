# Copyright 2026 the LlamaFactory team.
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

"""Stage-level trace hooks for diagnosing 4-node SFT hangs.

Imported once at process start (via tools/trace_train.py). Monkey-patches the
minimum granular stages that the cluster vendor asked for — dataloader_next /
forward / backward / optimizer_step / save_* — so when training hangs the last
`phase=enter` without a matching `phase=exit` pinpoints the stuck stage.

Each emitted line goes to stderr so the bash wrapper's `awk + tee` picks it up
with a millisecond timestamp. Format:

    [trace ts=<iso> rank=<R> pid=<P> step=<N> stage=<name> phase=<enter|exit|raise> dt_ms=<...>] <extra>

All patches are guarded; failure to patch any one target only logs a WARN and
training proceeds untouched.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable

_RANK = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
_PID = os.getpid()
_TRACE_COLLECTIVES = os.environ.get("TRACE_COLLECTIVES", "0") == "1"

_state_lock = threading.Lock()
_global_step = 0  # bumped on every Trainer.training_step entry


def _emit(stage: str, phase: str, dt_ms: float | None = None, extra: str = "") -> None:
    ts = datetime.now().isoformat(timespec="milliseconds")
    dt_part = f" dt_ms={dt_ms:.1f}" if dt_ms is not None else ""
    extra_part = f" {extra}" if extra else ""
    sys.stderr.write(
        f"[trace ts={ts} rank={_RANK} pid={_PID} step={_global_step} stage={stage} phase={phase}{dt_part}]{extra_part}\n"
    )
    sys.stderr.flush()


def _warn(msg: str) -> None:
    sys.stderr.write(f"[trace WARN rank={_RANK} pid={_PID}] {msg}\n")
    sys.stderr.flush()


def _wrap(stage: str, fn: Callable, extra_fn: Callable[..., str] | None = None) -> Callable:
    """Wrap fn so each call prints enter/exit/raise around the original body."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            extra = extra_fn(*args, **kwargs) if extra_fn else ""
        except Exception as exc:  # never let extra collection break training
            extra = f"extra_err={type(exc).__name__}"
        _emit(stage, "enter", extra=extra)
        t0 = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            dt = (time.monotonic() - t0) * 1000.0
            _emit(stage, "raise", dt_ms=dt, extra=f"exc={type(exc).__name__}")
            raise
        dt = (time.monotonic() - t0) * 1000.0
        _emit(stage, "exit", dt_ms=dt)
        return result

    return wrapper


# ---------- per-target install helpers ----------


def _install_dataloader() -> None:
    try:
        from torch.utils.data import dataloader as dl
    except Exception as exc:
        _warn(f"cannot import torch.utils.data.dataloader: {exc}")
        return

    def _next_extra(self) -> str:
        return f"loader_id={id(self):x}"

    try:
        dl._BaseDataLoaderIter.__next__ = _wrap(
            "dataloader_next", dl._BaseDataLoaderIter.__next__, _next_extra
        )
    except Exception as exc:
        _warn(f"cannot patch _BaseDataLoaderIter.__next__: {exc}")

    def _iter_extra(self) -> str:
        ds_len = "?"
        try:
            ds_len = str(len(self.dataset))  # type: ignore[arg-type]
        except Exception:
            pass
        return f"dataset_len={ds_len} num_workers={getattr(self, 'num_workers', '?')}"

    try:
        dl.DataLoader.__iter__ = _wrap(
            "dataloader_iter", dl.DataLoader.__iter__, _iter_extra
        )
    except Exception as exc:
        _warn(f"cannot patch DataLoader.__iter__: {exc}")


def _install_trainer() -> None:
    try:
        from transformers import Trainer
    except Exception as exc:
        _warn(f"cannot import transformers.Trainer: {exc}")
        return

    orig_training_step = Trainer.training_step

    @functools.wraps(orig_training_step)
    def training_step_wrapper(self, *args, **kwargs):
        global _global_step
        with _state_lock:
            _global_step = int(getattr(self.state, "global_step", _global_step)) + 1
        # extra: try to dig batch shape out of args
        extra = ""
        try:
            inputs = args[1] if len(args) >= 2 else kwargs.get("inputs")
            if isinstance(inputs, dict) and "input_ids" in inputs:
                shape = tuple(inputs["input_ids"].shape)
                extra = f"input_ids_shape={shape}"
        except Exception:
            pass
        _emit("training_step", "enter", extra=extra)
        t0 = time.monotonic()
        try:
            result = orig_training_step(self, *args, **kwargs)
        except BaseException as exc:
            _emit(
                "training_step",
                "raise",
                dt_ms=(time.monotonic() - t0) * 1000.0,
                extra=f"exc={type(exc).__name__}",
            )
            raise
        _emit("training_step", "exit", dt_ms=(time.monotonic() - t0) * 1000.0)
        return result

    try:
        Trainer.training_step = training_step_wrapper
    except Exception as exc:
        _warn(f"cannot patch Trainer.training_step: {exc}")

    def _save_ckpt_extra(self, *a, **kw) -> str:
        gs = getattr(getattr(self, "state", None), "global_step", "?")
        out = getattr(getattr(self, "args", None), "output_dir", "?")
        return f"global_step={gs} output_dir={out}"

    try:
        Trainer._save_checkpoint = _wrap(
            "save_checkpoint", Trainer._save_checkpoint, _save_ckpt_extra
        )
    except Exception as exc:
        _warn(f"cannot patch Trainer._save_checkpoint: {exc}")

    try:
        Trainer.save_model = _wrap("save_model", Trainer.save_model)
    except Exception as exc:
        _warn(f"cannot patch Trainer.save_model: {exc}")


def _install_compute_loss() -> None:
    """Patch Trainer.compute_loss — the de-facto forward boundary.

    HF Trainer.training_step calls `loss = self.compute_loss(model, inputs, ...)`,
    which is where `model(**inputs)` actually runs. We patch the base Trainer
    class because (a) Seq2SeqTrainer does not override compute_loss, and
    (b) LlamaFactory's CustomSeq2SeqTrainer.compute_loss falls through to
    `super().compute_loss(...)` on the non-ASFT path (the only path used by
    the qwen3_5 SFT yaml), so the base-class patch fires reliably.

    Patching `PreTrainedModel.forward` would be a no-op: forward is defined on
    each model subclass (e.g. Qwen3_5MoeForCausalLM), not on PreTrainedModel
    itself, so MRO resolves the subclass method and never sees the parent.
    """
    try:
        from transformers import Trainer
    except Exception as exc:
        _warn(f"cannot import Trainer for compute_loss patch: {exc}")
        return

    def _cl_extra(self, model, inputs, *a, **kw) -> str:
        try:
            ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
            if ids is not None and hasattr(ids, "shape"):
                return f"input_ids_shape={tuple(ids.shape)}"
        except Exception:
            pass
        return ""

    try:
        Trainer.compute_loss = _wrap("compute_loss", Trainer.compute_loss, _cl_extra)
    except Exception as exc:
        _warn(f"cannot patch Trainer.compute_loss: {exc}")


def _install_deepspeed() -> None:
    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except Exception as exc:
        _warn(f"cannot import DeepSpeedEngine (deepspeed not in use?): {exc}")
        return

    try:
        DeepSpeedEngine.backward = _wrap("backward", DeepSpeedEngine.backward)
    except Exception as exc:
        _warn(f"cannot patch DeepSpeedEngine.backward: {exc}")

    try:
        DeepSpeedEngine.step = _wrap("optimizer_step", DeepSpeedEngine.step)
    except Exception as exc:
        _warn(f"cannot patch DeepSpeedEngine.step: {exc}")

    def _save_extra(self, save_dir, tag=None, *a, **kw) -> str:
        return f"save_dir={save_dir} tag={tag}"

    try:
        DeepSpeedEngine.save_checkpoint = _wrap(
            "ds_save_checkpoint", DeepSpeedEngine.save_checkpoint, _save_extra
        )
    except Exception as exc:
        _warn(f"cannot patch DeepSpeedEngine.save_checkpoint: {exc}")

    if hasattr(DeepSpeedEngine, "_save_zero_checkpoint"):
        try:
            DeepSpeedEngine._save_zero_checkpoint = _wrap(
                "ds_save_zero_checkpoint", DeepSpeedEngine._save_zero_checkpoint
            )
        except Exception as exc:
            _warn(f"cannot patch DeepSpeedEngine._save_zero_checkpoint: {exc}")


def _install_save_pretrained() -> None:
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception as exc:
        _warn(f"cannot import PreTrainedModel for save_pretrained: {exc}")
        return

    def _sp_extra(self, save_directory, *a, **kw) -> str:
        return f"save_dir={save_directory}"

    try:
        PreTrainedModel.save_pretrained = _wrap(
            "save_pretrained", PreTrainedModel.save_pretrained, _sp_extra
        )
    except Exception as exc:
        _warn(f"cannot patch PreTrainedModel.save_pretrained: {exc}")


def _install_safetensors() -> None:
    try:
        import safetensors.torch as st
    except Exception as exc:
        _warn(f"cannot import safetensors.torch: {exc}")
        return

    def _sf_extra(tensors, filename, *a, **kw) -> str:
        try:
            n = len(tensors) if hasattr(tensors, "__len__") else "?"
        except Exception:
            n = "?"
        return f"file={filename} n_tensors={n}"

    try:
        st.save_file = _wrap("safetensors_save_file", st.save_file, _sf_extra)
    except Exception as exc:
        _warn(f"cannot patch safetensors.torch.save_file: {exc}")


def _install_collectives() -> None:
    if not _TRACE_COLLECTIVES:
        return
    try:
        import torch.distributed as dist
    except Exception as exc:
        _warn(f"cannot import torch.distributed: {exc}")
        return

    for name in ("barrier", "all_reduce", "broadcast", "_reduce_scatter_base", "all_gather"):
        fn = getattr(dist, name, None)
        if fn is None:
            continue
        try:
            setattr(dist, name, _wrap(f"coll_{name}", fn))
        except Exception as exc:
            _warn(f"cannot patch torch.distributed.{name}: {exc}")


# ---------- entry ----------


def install() -> None:
    _install_dataloader()
    _install_trainer()
    _install_compute_loss()
    _install_deepspeed()
    _install_save_pretrained()
    _install_safetensors()
    _install_collectives()
    _emit("trace_hook", "installed", extra=f"collectives={_TRACE_COLLECTIVES}")


install()

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

import json
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from mcore_adapter.trainer import McaTrainer

from ...extras import logging
from ...extras.constants import IGNORE_INDEX


logger = logging.get_logger(__name__)


class CustomMcaTrainer(McaTrainer):
    def _pre_compute_loss(self, data_iterator, model):
        compute = super()._pre_compute_loss
        if not getattr(self, "_profile_active", False):
            return compute(data_iterator, model)
        self._profile_microbatch += 1
        index = self._profile_microbatch
        if self.is_world_process_zero() and (index == 1 or index % 8 == 0):
            logger.info(
                f"[perf] step={self.state.global_step + 1} forward microbatch="
                f"{index}/{self.args.gradient_accumulation_steps} "
                f"elapsed={time.perf_counter() - self._profile_start:.1f}s"
            )
        return compute(data_iterator, model)

    def training_step(self, *args, **kwargs):
        if os.environ.get("LF_MCA_PROFILE", "0") != "1":
            return super().training_step(*args, **kwargs)

        def synchronize():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        timings = {}

        def timed(name, function):
            def call(*call_args, **call_kwargs):
                synchronize()
                start = time.perf_counter()
                result = function(*call_args, **call_kwargs)
                synchronize()
                timings[name] = time.perf_counter() - start
                return result

            return call

        synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._profile_start = time.perf_counter()
        self._profile_microbatch = 0
        self._profile_active = True
        forward_backward = self.forward_backward_func
        optimizer_step = self.optimizer.step
        self.forward_backward_func = timed("forward_backward_seconds", forward_backward)
        self.optimizer.step = timed("optimizer_seconds", optimizer_step)
        try:
            result = super().training_step(*args, **kwargs)
        finally:
            self.forward_backward_func = forward_backward
            self.optimizer.step = optimizer_step
            self._profile_active = False
        timings["training_step_seconds"] = time.perf_counter() - self._profile_start
        if torch.cuda.is_available():
            timings["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            timings["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            # All ranks call once after each successful step. Report maxima,
            # since rank 0 alone may not host the stage with the memory peak.
            if torch.distributed.is_initialized():
                values = torch.tensor(list(timings.values()), dtype=torch.float64, device=self.args.device)
                torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
                timings = dict(zip(timings, values.tolist()))
        self._profile_metrics = timings
        if self.is_world_process_zero():
            logger.info("[perf] " + json.dumps({"step": self.state.global_step + 1, **timings}))
        return result

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        logs = dict(kwargs.get("other_logs", {}))
        if getattr(self, "_profile_metrics", None):
            logs.update(self._profile_metrics)
        kwargs["other_logs"] = logs
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    @override
    def _pad_batched_inputs(self, inputs: dict[str, Tensor | Any], seq_length: int):
        r"""Override to avoid padding error when handling 3d posids."""
        padding_inputs = {
            k: v.tolist() if v is not None and isinstance(v, Tensor) else v
            for k, v in inputs.items()
            if k in self._language_input_names
        }

        position_ids_3d = None
        if isinstance(inputs.get("position_ids"), Tensor) and inputs["position_ids"].dim() == 3:
            position_ids_3d = inputs["position_ids"]
            padding_inputs.pop("position_ids", None)

        tokenizer = (
            self.processing_class
            if isinstance(self.processing_class, PreTrainedTokenizerBase)
            else getattr(self.processing_class, "tokenizer", self.processing_class)
        )
        padding_side = getattr(tokenizer, "padding_side", "right")
        if "labels" in padding_inputs:
            rows = padding_inputs["labels"]
            if any(len(row) > seq_length for row in rows):
                raise ValueError("Cannot pad labels to a shorter sequence.")
            padding_inputs["labels"] = [
                [IGNORE_INDEX] * (seq_length - len(row)) + row
                if padding_side == "left"
                else row + [IGNORE_INDEX] * (seq_length - len(row))
                for row in rows
            ]
        padding_inputs = tokenizer.pad(
            padding_inputs,
            padding="max_length",
            max_length=seq_length,
            return_tensors="pt",
        ).to(self.args.device)
        inputs.update(padding_inputs)
        if "loss_weights" in inputs:
            pad_len = seq_length - inputs["loss_weights"].size(-1)
            if pad_len < 0:
                raise ValueError("Cannot pad loss_weights to a shorter sequence.")
            pad = (pad_len, 0) if padding_side == "left" else (0, pad_len)
            inputs["loss_weights"] = F.pad(inputs["loss_weights"], pad, value=0).to(self.args.device)

        if position_ids_3d is not None:
            current_seq_len = position_ids_3d.size(-1)
            if current_seq_len < seq_length:
                pad_len = seq_length - current_seq_len
                if padding_side == "left":
                    position_ids_3d = F.pad(position_ids_3d, (pad_len, 0), value=0)
                else:
                    position_ids_3d = F.pad(position_ids_3d, (0, pad_len), value=0)

            inputs["position_ids"] = position_ids_3d.to(self.args.device)

        return inputs

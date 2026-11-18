# Copyright 2026 the LlamaFactory team. Licensed under the Apache License, Version 2.0.
"""Opt-in BF16/FSDP2 sharded optimizer restore for Accelerate 1.11.0.

Use named, initialized optimizer-state placeholders when loading DCP checkpoints.
"""

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


class FSDP2ResumeMixin:
    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return super()._load_optimizer_and_scheduler(checkpoint)
        path = Path(checkpoint) / "optimizer_0"
        if not getattr(self, "is_fsdp_enabled", False) or not self.accelerator.is_fsdp2 or not path.is_dir():
            return super()._load_optimizer_and_scheduler(checkpoint)
        if self.accelerator.scaler is not None:
            raise ValueError("This validated restore path requires BF16/no GradScaler")
        opt = self.optimizer.optimizer if hasattr(self.optimizer, "optimizer") else self.optimizer
        options = StateDictOptions(full_state_dict=False, cpu_offload=False)
        state = {"optimizer": get_optimizer_state_dict(self.model, opt, options=options)}
        dcp.load(state, checkpoint_id=path)
        set_optimizer_state_dict(self.model, opt, state["optimizer"], options=options)
        self.lr_scheduler.load_state_dict(
            torch.load(Path(checkpoint) / "scheduler.pt", map_location="cpu", weights_only=True)
        )
        self.accelerator.wait_for_everyone()

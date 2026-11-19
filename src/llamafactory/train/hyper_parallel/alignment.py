"""Explicit, auditable sample-mean alignment with the supplied GPU experiment."""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler
from transformers import TrainerCallback


class PaddedSequentialSampler(SequentialSampler):
    """Sequential epochs padded to whole global batches, rotating padding rows."""

    def __init__(self, data_source, multiple):
        super().__init__(data_source)
        self.multiple = multiple
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.data_source) / self.multiple) * self.multiple

    def __iter__(self):
        size = len(self.data_source)
        if not size:
            return iter(())
        padding = len(self) - size
        start = self.epoch * padding % size
        return iter(list(range(size)) + [(start + offset) % size for offset in range(padding)])

    def set_epoch(self, epoch):
        self.epoch = epoch


class AlignmentCallback(TrainerCallback):
    """Keep the production LR schedule when bounding a validation run."""

    def __init__(self, validation_steps):
        self.validation_steps = validation_steps
        self.started = None
        self.last_step_time = None
        self.last_step_seconds = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = self.last_step_time = time.monotonic()
        if not self.validation_steps:
            return
        model = kwargs["model"]
        directory = Path("/cache/qwen38-alignment/trace") / Path(args.output_dir).name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank-{args.process_index}.json"
        def record(index, phase):
            def hook(*unused):
                if state.global_step > 0:
                    return
                payload = {"time": time.time(), "pid": os.getpid(), "rank": args.process_index,
                    "layer": index, "phase": phase, "step": state.global_step,
                    "allocated_gib": torch.npu.memory_allocated() / 2**30,
                    "reserved_gib": torch.npu.memory_reserved() / 2**30,
                    "peak_allocated_gib": torch.npu.max_memory_allocated() / 2**30}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload))
                temporary.replace(path)
            return hook
        for index, layer in enumerate(model.model.language_model.layers):
            layer.register_forward_pre_hook(record(index, "enter"))
            layer.register_forward_hook(record(index, "exit"))

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        self.last_step_seconds = now - self.last_step_time
        self.last_step_time = now
        if self.validation_steps > 0 and state.global_step >= self.validation_steps:
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        values = dict(logs or {})
        values["timestamp"] = time.time()
        if self.started is not None:
            values["elapsed_training_seconds"] = time.monotonic() - self.started
        if "loss" in values:
            values["step_seconds"] = self.last_step_seconds
        if 'learning_rate' in values:
            values['learning_rate_used'] = values['learning_rate']
            values['learning_rate_after_update'] = kwargs['optimizer'].param_groups[0]['lr']
        if hasattr(torch, 'npu'):
            values['peak_npu_allocated_gib_rank0'] = torch.npu.max_memory_allocated() / 2**30
        path = Path(args.output_dir) / 'alignment-metrics.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps({'step': state.global_step, **values}) + '\n')


def audit_microbatch(trainer, inputs, scaled_loss):
    """Record hashes rather than raw user text so token/mask order is inspectable."""
    length = int(inputs['attention_mask'][0].sum().item())
    ids = inputs['input_ids'][0, :length].detach().cpu().numpy().astype('<i8')
    labels = inputs['labels'][0, :length].detach().cpu().numpy().astype('<i8')
    accumulation = getattr(trainer, 'current_gradient_accumulation_steps', trainer.args.gradient_accumulation_steps)
    row = {
        'micro_step': trainer._alignment_micro_step,
        'optimizer_step_before_update': trainer.state.global_step,
        'rank': trainer.args.process_index,
        'cp_size': trainer._cp_size,
        'cp_rank': trainer._cp_mesh.get_local_rank() if trainer._cp_size > 1 else 0,
        'length': length,
        'supervised_tokens': int(np.count_nonzero(labels[1:] != -100)),
        'input_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
        'sample_loss': float(scaled_loss.item()) * accumulation,
    }
    path = Path(trainer.args.output_dir) / f'micro-rank{trainer.args.process_index}.jsonl'
    with path.open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    trainer._alignment_micro_step += 1

# Copyright 2026 the LlamaFactory team. Licensed under the Apache License, Version 2.0.
"""Opt-in GPU sample-order/loss semantics for CP=1 FSDP2, microbatch=1.

Run with accelerate launch --config_file accelerate.yaml aligned_train.py config.yaml.
Does not claim optimizer numerical equivalence to Megatron precision-aware Adam.
"""

import math
import os


os.environ.setdefault("IMAGE_PLACEHOLDER", "<|audit_image_placeholder|>")
os.environ.setdefault("VIDEO_PLACEHOLDER", "<|audit_video_placeholder|>")
os.environ.setdefault("AUDIO_PLACEHOLDER", "<|audit_audio_placeholder|>")
from torch.utils.data import SequentialSampler

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


class PaddedSequentialSampler(SequentialSampler):
    def __init__(self, data_source, multiple=64):
        super().__init__(data_source)
        self.multiple = multiple
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.data_source) / self.multiple) * self.multiple

    def __iter__(self):
        n = len(self.data_source)
        if not n:
            return iter(())
        pad = len(self) - n
        start = (self.epoch * pad) % n
        return iter(list(range(n)) + [(start + i) % n for i in range(pad)])

    def set_epoch(self, epoch):
        self.epoch = epoch


from fsdp2_resume import FSDP2ResumeMixin  # noqa: E402 - initialize environment before trainer imports


class GPUAlignedTrainer(FSDP2ResumeMixin, CustomSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("Alignment wrapper requires microbatch=1")
        self.global_batch = self.args.gradient_accumulation_steps * self.accelerator.num_processes
        if os.getenv("MCA_EPOCH_SHUFFLE_SEED"):
            raise ValueError("This wrapper reproduces the non-shuffled GPU sampler only")
        self.model_accepts_loss_kwargs = False

    def get_decay_parameter_names(self, model):
        # Megatron-Core 0.18.2 default: exempt all one-dimensional parameters and biases.
        return [name for name, param in model.named_parameters() if param.ndim != 1 and not name.endswith(".bias")]

    def _get_train_sampler(self, train_dataset=None):
        return PaddedSequentialSampler(
            self.train_dataset if train_dataset is None else train_dataset, self.global_batch
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # With microbatch=1, normal model mean CE is the required per-sample token mean.
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=None)


if __name__ == "__main__":
    import llamafactory.train.sft.workflow as workflow
    from llamafactory.train.tuner import run_exp

    workflow.CustomSeq2SeqTrainer = GPUAlignedTrainer
    run_exp()

"""Bound whole-layer recomputation by checkpointing expensive submodules."""

import functools
import os

import torch
from memory import checkpoint


def install_nested_checkpoint(model):
    count = 0

    def wrap(module):
        original = module.forward

        @functools.wraps(original)
        def forward(*args, **kwargs):
            if not torch.is_grad_enabled():
                return original(*args, **kwargs)
            return checkpoint(original, *args, use_reentrant=False, **kwargs)

        module.forward = forward

    for layer in model.model.language_model.layers:
        names = ["input_layernorm", "post_attention_layernorm", "mlp"]
        if os.getenv("PROBE_RECOMPUTE_OFFLOAD") != "1":
            names += ["self_attn", "linear_attn"]
        for name in names:
            module = getattr(layer, name, None)
            if module is not None:
                wrap(module)
                count += 1
        if os.getenv("PROBE_RECOMPUTE_OFFLOAD") == "1":
            continue
        attn = getattr(layer, "self_attn", getattr(layer, "linear_attn", None))
        for name in ["q_norm", "k_norm", "norm"]:
            module = getattr(attn, name, None)
            if module is not None:
                wrap(module)
                count += 1
    return count

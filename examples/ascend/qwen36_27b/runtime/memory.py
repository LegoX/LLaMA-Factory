"""Opt-in native PyTorch memory controls for text-only alignment probes."""

import functools
import os

import torch
from torch.utils.checkpoint import checkpoint


def chunked_ce(hidden, labels, head, chunk_size=256):
    # HF causal shift and ignore_index=-100, mean over supervised tokens per sample.
    h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    y = labels[:, 1:].reshape(-1)
    selected = torch.nonzero(y != -100).flatten()
    if not selected.numel():
        raise ValueError("No supervised tokens after truncation")

    def part(x, target):
        return torch.nn.functional.cross_entropy(head(x).float(), target, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for ids in selected.split(chunk_size):
        total = total + checkpoint(part, h.index_select(0, ids), y.index_select(0, ids), use_reentrant=False)
    return total / selected.numel()


class ChunkedLossHead(torch.nn.Linear):
    def forward(self, hidden, labels=None):
        if labels is None:
            return torch.nn.functional.linear(hidden, self.weight, self.bias)
        from linear_ce import linear_ce

        return linear_ce(hidden, labels, self)


def patch_chunked_ce(model):
    if os.getenv("PROBE_DISABLE_BACKWARD_PREFETCH") == "1":
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

        def no_backward_prefetch(group):
            # Replay appends forward-order entries; avoid gathering already consumed groups.
            if group._post_forward_indices:
                group._post_forward_indices.pop()

        FSDPParamGroup._backward_prefetch = no_backward_prefetch
    if os.getenv("PROBE_SHARD_LOSS_HEAD") == "1":
        assert isinstance(model.lm_head, torch.nn.Linear) and model.lm_head.bias is None
        assert not model.config.tie_word_embeddings, "Separate head sharding requires untied weights"
        model.lm_head.__class__ = ChunkedLossHead
    original = model.forward

    @functools.wraps(original)
    def forward(input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):
        if labels is None:
            return original(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

        for key in ["num_items_in_batch", "logits_to_keep", "return_dict"]:
            kwargs.pop(key, None)
        # Preserve multimodal inputs: LF injects masked dummy images for text-only FSDP batches.
        if os.getenv("PROBE_TEXT_ONLY") == "1":
            # All training rows are text. LF's appended dummy image is masked and unsupervised.
            assert input_ids.shape[0] == 1 and attention_mask.ndim == 2
            length = int(attention_mask[0].sum())
            assert bool(attention_mask[0, :length].all()) and not bool(attention_mask[0, length:].any())
            valid = input_ids[:, :length]
            assert not bool(((valid == model.config.image_token_id) | (valid == model.config.video_token_id)).any()), (
                "Actual media tokens in text-only path"
            )
            assert not bool((labels[:, length:] != -100).any()), "Attempted to remove supervised tokens"
            input_ids = valid
            labels = labels[:, :length]
            attention_mask = attention_mask[:, :length]
            if position_ids is not None:
                position_ids = position_ids[..., :length]
            for key in [
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "mm_token_type_ids",
                "rope_deltas",
                "second_per_grid_ts",
            ]:
                kwargs.pop(key, None)
            outputs = model.model.language_model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs
            )
        else:
            outputs = model.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs
            )
        if os.getenv("PROBE_FP32_CE") == "1":
            from linear_ce import linear_ce

            loss = (
                model.lm_head(outputs[0], labels)
                if os.getenv("PROBE_SHARD_LOSS_HEAD") == "1"
                else linear_ce(outputs[0], labels, model.lm_head)
            )
        else:
            loss = chunked_ce(outputs[0], labels, model.lm_head)
        return Qwen3_5CausalLMOutputWithPast(loss=loss, logits=None)

    model.forward = forward


def offload_checkpoint(function, *args, **kwargs):
    # Checkpoint's own inner hooks discard intermediates; outer hooks move retained
    # checkpoint inputs to CPU and restore them to their original device for replay.
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        return checkpoint(function, *args, **kwargs)

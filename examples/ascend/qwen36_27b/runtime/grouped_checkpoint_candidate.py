"""Isolated grouped decoder checkpoint candidate; retains layer/FSDP modules."""

import contextlib
import inspect
import os
import textwrap
import types

from memory import offload_checkpoint
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf


def install_grouped_checkpoint(model, group_size=2):
    text_model = model.model.language_model
    source = textwrap.dedent(inspect.getsource(hf.Qwen3_5TextModel.forward))
    source = source.replace("@auto_docstring\n", "")  # Documentation decorator assumes a class-qualified method.
    start = source.index("    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):")
    end = source.index("    hidden_states = self.norm(hidden_states)", start)
    replacement = (
        """    if self.training and torch.is_grad_enabled():
        if use_cache or past_key_values is not None:
            raise ValueError("Grouped training checkpoint requires cache disabled")
        size=self._checkpoint_group_size
        for start in range(0,self.config.num_hidden_layers,size):
            group=tuple(self.layers[start:min(start+size,self.config.num_hidden_layers)])
            masks=tuple(linear_attn_mask if self.config.layer_types[start+j]=="linear_attention" else causal_mask for j in range(len(group)))
            for layer in group:layer.gradient_checkpointing=False
            def run_group(hidden, group=group, masks=masks):
                with torch.autograd.graph.save_on_cpu(pin_memory=True) if self._group_reentrant else contextlib.nullcontext():
                    for layer,mask in zip(group,masks):
                        hidden=layer(hidden,position_embeddings=position_embeddings,attention_mask=mask,
                                     position_ids=text_position_ids,past_key_values=None,use_cache=False,**kwargs)
                    return hidden
            hidden_states=offload_checkpoint(run_group,hidden_states,use_reentrant=self._group_reentrant)
            self._group_checkpoint_calls+=1
    else:
"""
        + "".join("    " + line + "\n" for line in source[start:end].splitlines())
        + "\n"
    )
    source = source[:start] + replacement + source[end:]
    namespace = dict(vars(hf))
    namespace["offload_checkpoint"] = offload_checkpoint
    namespace["contextlib"] = contextlib
    exec(compile(source, "<grouped_qwen35_checkpoint>", "exec"), namespace)
    text_model.forward = types.MethodType(namespace["forward"], text_model)
    text_model._group_reentrant = os.getenv("PROBE_GROUP_REENTRANT") == "1"
    text_model._checkpoint_group_size = group_size
    text_model._group_checkpoint_calls = 0

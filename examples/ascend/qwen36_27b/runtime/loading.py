"""Avoid redundant initialization of fully loaded Qwen3.5 leaf parameters.

Transformers 5.8's Linear init calls weight.float(), losing the initialized marker
on a temporary copy for BF16 weights. Skip only modules whose direct parameters
and buffers are all explicitly marked as loaded; missing buffers still initialize.
"""


def install_loaded_leaf_guard():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

    original = Qwen3_5PreTrainedModel._initialize_weights
    if getattr(original, "_loaded_leaf_guard", False):
        return

    def initialize(self, module, is_remote_code=False):
        params = list(module.parameters(recurse=False))
        buffers = list(module.buffers(recurse=False))
        if params and all(getattr(p, "_is_hf_initialized", False) for p in params + buffers):
            module._is_hf_initialized = True
            return
        return original(self, module, is_remote_code=is_remote_code)

    initialize._loaded_leaf_guard = True
    Qwen3_5PreTrainedModel._initialize_weights = initialize


def verify_loaded_weights(model, output_path):
    import collections
    import json
    import pathlib

    import torch
    from safetensors import safe_open

    source = pathlib.Path(model.config._name_or_path)
    index = source / "model.safetensors.index.json"
    state = model.state_dict()
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(source / "model.safetensors", framework="pt", device="cpu") as f:
            weight_map = dict.fromkeys(f.keys(), "model.safetensors")
    groups = collections.defaultdict(list)
    for name in state:
        if name not in weight_map:
            raise ValueError(f"Model tensor absent from checkpoint: {name}")
        groups[weight_map[name]].append(name)
    checked = 0
    for filename, names in groups.items():
        with safe_open(source / filename, framework="pt", device="cpu") as f:
            for name in names:
                actual = state[name].detach()
                expected = f.get_tensor(name).to(dtype=actual.dtype, device=actual.device)
                if not torch.equal(actual, expected):
                    raise ValueError(f"Loaded weight mismatch: {name}")
                checked += 1
    pathlib.Path(output_path).write_text(
        json.dumps(
            {
                "source": str(source),
                "tensors_checked": checked,
                "exact_after_dtype_conversion": True,
                "unused_checkpoint_keys": sorted(set(weight_map) - set(state)),
            },
            indent=2,
        )
    )

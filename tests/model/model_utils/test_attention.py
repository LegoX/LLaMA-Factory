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

import os

import pytest
from transformers.utils import is_flash_attn_2_available


# Compatible with Transformers v4 and Transformers v5
try:
    from transformers.utils import is_torch_sdpa_available
except ImportError:

    def is_torch_sdpa_available():
        return True


from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.train.test_utils import load_infer_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "template": "llama3",
}


def test_flash_attention_2_s_aux_none_wrapper(monkeypatch):
    import torch
    from types import SimpleNamespace

    import transformers.modeling_flash_attention_utils as flash_utils

    from llamafactory.model.model_utils.attention import _flash_attention_forward_with_optional_s_aux

    captured = {}

    def fake_flash_attention_forward(query, key, value, attention_mask, **kwargs):
        captured["s_aux"] = kwargs["s_aux"]
        return torch.zeros_like(query)

    monkeypatch.setattr(flash_utils, "_flash_attention_forward", fake_flash_attention_forward)
    module = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        is_causal=False,
        layer_idx=0,
    )
    states = torch.zeros((1, 2, 3, 4), dtype=torch.float16)

    output, weights = _flash_attention_forward_with_optional_s_aux(module, states, states, states, None, s_aux=None)

    assert captured["s_aux"] is None
    assert output.shape == (1, 3, 2, 4)
    assert weights is None


@pytest.mark.xfail(is_transformers_version_greater_than("4.48"), reason="Attention refactor.")
def test_attention():
    attention_available = ["disabled"]
    if is_torch_sdpa_available():
        attention_available.append("sdpa")

    if is_flash_attn_2_available():
        attention_available.append("fa2")

    llama_attention_classes = {
        "disabled": "LlamaAttention",
        "sdpa": "LlamaSdpaAttention",
        "fa2": "LlamaFlashAttention2",
    }
    for requested_attention in attention_available:
        model = load_infer_model(flash_attn=requested_attention, **INFER_ARGS)
        for module in model.modules():
            if "Attention" in module.__class__.__name__:
                assert module.__class__.__name__ == llama_attention_classes[requested_attention]

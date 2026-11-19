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

"""Validate the shared FSDP2 policy against dense and MoE Qwen3.5 classes."""

from pathlib import Path

import pytest
import torch
import yaml
from accelerate import FullyShardedDataParallelPlugin


modeling_qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
modeling_qwen3_5_moe = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")


@pytest.mark.parametrize("module, prefix", [(modeling_qwen3_5, "Qwen3_5"), (modeling_qwen3_5_moe, "Qwen3_5Moe")])
def test_shared_fsdp_config_infers_actual_transformer_classes(module, prefix, monkeypatch):
    monkeypatch.delenv("FSDP_TRANSFORMER_CLS_TO_WRAP", raising=False)
    path = Path(__file__).resolve().parents[2] / "examples/accelerate/fsdp2_config_qwen35.yaml"
    settings = yaml.safe_load(path.read_text())["fsdp_config"]
    model_cls = getattr(module, prefix + "ForConditionalGeneration")
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    # Keep the real model/layer types without allocating any weights.
    for name in model._no_split_modules:
        layer_cls = getattr(module, name)
        layer = layer_cls.__new__(layer_cls)
        torch.nn.Module.__init__(layer)
        model.add_module(name, layer)
    explicit_classes = settings.get("fsdp_transformer_layer_cls_to_wrap")
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy=settings["fsdp_auto_wrap_policy"],
        transformer_cls_names_to_wrap=explicit_classes.split(",") if explicit_classes else None,
    )
    plugin.set_auto_wrap_policy(model)
    actual = {cls.__name__ for cls in plugin.auto_wrap_policy.keywords["transformer_layer_cls"]}
    assert actual == set(model._no_split_modules)

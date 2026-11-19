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

"""Exercise the real save/load entry points without importing the CUDA model stack."""

import ast
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import pytest


@pytest.fixture
def model_factory():
    path = Path(__file__).resolve().parents[2] / "src/mcore_adapter/models/model_factory.py"
    tree = ast.parse(path.read_text())
    guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_direct_hf_save")
    methods = []
    for class_name, method_name in [
        ("PretrainedModel", "from_pretrained"),
        ("VirtualModels", "save_pretrained_as_hf"),
    ]:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
        method.decorator_list = []
        methods.append(method)
    ns = dict(os=os, time=time, Union=Union, MAX_SHARD_SIZE="5GB", PreTrainedTokenizer=object)
    exec(compile(ast.Module(body=[guard, *methods], type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("mode", ["rotary_dimensions", "all_dimensions"])
@pytest.mark.parametrize("save_hf", [False, True])
def test_yarn_save_choice_checked_on_resolved_config_before_model_allocation(model_factory, mode, save_hf):
    config = SimpleNamespace(mrope_yarn_enabled=True, mrope_yarn_mscale_mode=mode)
    # Represents config resolved from the source checkpoint plus training overrides.
    cls = SimpleNamespace(config_class=SimpleNamespace(from_pretrained=lambda *args: config))

    class AllocationReached(Exception):
        pass

    def allocate(*args, **kwargs):
        raise AllocationReached

    model_factory["VirtualModels"] = allocate
    expected = ValueError if save_hf else AllocationReached
    with pytest.raises(expected) as result:
        model_factory["from_pretrained"](cls, "source-checkpoint", SimpleNamespace(save_hf_model=save_hf))
    if save_hf:
        assert "save_hf_model=False" in str(result.value)


@pytest.mark.parametrize("mode", ["rotary_dimensions", "all_dimensions"])
def test_direct_yarn_save_rejected_before_creating_output(model_factory, tmp_path, mode):
    model = SimpleNamespace(config=SimpleNamespace(mrope_yarn_enabled=True, mrope_yarn_mscale_mode=mode))
    output = tmp_path / "hf"
    with pytest.raises(ValueError, match="megatron_export"):
        model_factory["save_pretrained_as_hf"](model, str(output))
    assert not output.exists()


def test_prevalidated_config_is_reused_for_model_allocation(model_factory):
    config = SimpleNamespace(mrope_yarn_enabled=False)
    cls = SimpleNamespace(config_class=SimpleNamespace(from_pretrained=lambda *a: pytest.fail("config reloaded")))

    class AllocationReached(Exception):
        pass

    def allocate(model_cls, **kwargs):
        assert model_cls is cls and kwargs["config"] is config
        raise AllocationReached

    model_factory["VirtualModels"] = allocate
    with pytest.raises(AllocationReached):
        model_factory["from_pretrained"](cls, "checkpoint", config=config)


@pytest.mark.parametrize("folding", [False, True])
def test_native_context_direct_hf_save_still_reaches_converter(model_factory, tmp_path, folding):
    config = SimpleNamespace(mrope_yarn_enabled=False, moe_parallel_folding=folding)
    model = SimpleNamespace(config=config, models=[object()])
    calls = []
    converter = SimpleNamespace(
        save_model_as_hf_inflight=lambda *a, **kw: calls.append(("standard", a, kw)),
        save_separate_model_as_hf_inflight=lambda *a, **kw: calls.append(("folded", a, kw)),
    )
    model_factory["ModelConverter"] = lambda *a, **kw: converter
    output = tmp_path / "hf"
    model_factory["save_pretrained_as_hf"](model, str(output))
    assert output.is_dir() and len(calls) == 1
    route, args, kwargs = calls[0]
    assert route == ("folded" if folding else "standard")
    assert args == (model.models, str(output)) and kwargs["save_safetensors"]

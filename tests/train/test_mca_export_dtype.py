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

"""Exercise the real streaming writers and metadata finalizer on tiny CPU tensors."""

import ast
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from safetensors.torch import load_file, save_file


Qwen3_5Config = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5").Qwen3_5Config
SOURCE = Path(__file__).resolve().parents[2] / "src/mcore_adapter/models/converter/post_converter.py"


@pytest.mark.parametrize("converter", ["StreamingHFConverter", "DistStreamingHFConverter"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, None])
def test_streaming_export_dtype_and_metadata(tmp_path, converter, dtype):
    tree = ast.parse(SOURCE.read_text())
    methods = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            names = (
                {"__init__", "_finalize"}
                if cls.name == "BaseHFConverter"
                else {"convert"}
                if cls.name == converter
                else set()
            )
            methods.extend(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names)
    tokenizer = SimpleNamespace(save_pretrained=lambda directory: None)
    namespace = dict(
        torch=torch,
        os=os,
        gc=gc,
        json=json,
        Optional=Optional,
        save_file=save_file,
        tqdm=lambda items, **kw: items,
        logger=SimpleNamespace(info=lambda *a: None),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    cls = type("Converter", (), {method.name: namespace[method.name] for method in methods})
    obj = cls.__new__(cls)
    obj.mca_config = SimpleNamespace(params_dtype=torch.bfloat16, save_hf_auto_map_files=lambda directory: None)
    obj.hf_config = Qwen3_5Config(dtype=torch.float16, text_config={"dtype": "float16"})
    obj._setup = lambda: None
    obj.__init__("unused", str(tmp_path), dtype, False)
    source = {
        "model.weight": torch.tensor([1.234567, 2.345678, 3.456789]),
        "model.indices": torch.tensor([0, 1, 2]),
    }
    obj.max_shard_bytes = 5
    obj._stream_hf_weights = lambda *a, **kw: iter(source.items())
    obj._init_model = lambda: SimpleNamespace(sharded_state_dict=lambda: {})
    obj._stream_dist_weights_from_shards = lambda *a, **kw: iter(source.items())
    obj.convert()
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    actual = {}
    for filename in set(index["weight_map"].values()):
        actual.update(load_file(tmp_path / filename))
    expected_dtype = dtype or torch.bfloat16
    torch.testing.assert_close(actual["model.weight"], source["model.weight"].to(expected_dtype))
    torch.testing.assert_close(actual["model.indices"], source["model.indices"])
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in actual.values())
    config = Qwen3_5Config.from_pretrained(tmp_path, local_files_only=True)
    assert config.dtype == config.text_config.dtype == config.vision_config.dtype == expected_dtype

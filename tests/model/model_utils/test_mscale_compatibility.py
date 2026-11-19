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

import ast
import base64
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


pytest.importorskip("fcntl", reason="Legacy mscale compensation uses POSIX file locks")

from llamafactory.model.model_utils.legacy_mscale import compensate_legacy_mscale  # noqa: E402


@pytest.fixture
def rotary_functions(megatron_distribution):
    source = Path(__file__).resolve().parents[3] / "src/mcore_adapter/models/qwen3_vl/rope_utils.py"
    if not source.exists():
        pytest.skip("Local adapter checkout is unavailable")
    mega = Path(megatron_distribution.locate_file("megatron/core/models/common/embeddings/rope_utils.py"))
    ns = {"torch": torch, "Tensor": torch.Tensor, "Optional": Optional, "Qwen3VLConfig": SimpleNamespace}
    for path, names in [
        (mega, {"_rotate_half", "_apply_rotary_pos_emb_bshd"}),
        (source, {"_apply_rotary_pos_emb_thd_absolute", "apply_rotary_pos_emb_absolute"}),
    ]:
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("yarn", [False, True])
def test_corrected_forward_and_backward_preserve_nonrotary_dimensions(rotary_functions, packed, yarn):
    ns = rotary_functions
    torch.manual_seed(42)
    x = torch.randn((7, 3, 256) if packed else (7, 2, 3, 256), requires_grad=True)
    freqs = torch.randn(7, 1 if packed else 2, 1, 64)
    config = SimpleNamespace(rotary_interleaved=False, mrope_yarn_enabled=yarn)
    scale = 1 + 0.1 * math.log(2)
    result = ns["apply_rotary_pos_emb_absolute"](
        x, freqs, config, cu_seqlens=torch.tensor([0, 3, 7]) if packed else None, mscale=scale
    )
    expected = ns["_apply_rotary_pos_emb_bshd"](x[:, None] if packed else x, freqs, mscale=scale if yarn else 1.0)
    if packed:
        expected = expected.squeeze(1)
    assert torch.equal(result, expected)
    assert torch.equal(result[..., 64:], x[..., 64:])
    result.sum().backward()
    assert torch.equal(x.grad[..., 64:], torch.ones_like(x.grad[..., 64:]))


@pytest.mark.parametrize("packed", [False, True])
def test_explicit_legacy_mode_preserves_old_computation(rotary_functions, packed):
    ns = rotary_functions
    x = torch.randn((7, 3, 256) if packed else (7, 1, 3, 256))
    freqs = torch.randn(7, 1, 1, 64)
    scale = 1 + 0.1 * math.log(2)
    config = SimpleNamespace(
        rotary_interleaved=False, mrope_yarn_enabled=True, mrope_yarn_mscale_mode="all_dimensions"
    )
    result = ns["apply_rotary_pos_emb_absolute"](
        x, freqs, config, mscale=scale, cu_seqlens=torch.tensor([0, 3, 7]) if packed else None
    )
    expected = ns["_apply_rotary_pos_emb_bshd"](x[:, None] if packed else x, freqs) * scale
    assert torch.equal(result, expected.squeeze(1) if packed else expected)


def test_norm_reparameterization_is_equivalent_in_real_arithmetic(rotary_functions):
    ns = rotary_functions
    x = torch.randn(7, 1, 3, 256, dtype=torch.float64)
    weight = torch.randn(256, dtype=torch.float64) * 0.1
    normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    scale = 1 + 0.1 * math.log(2)
    freqs = torch.randn(7, 1, 1, 64, dtype=torch.float64)
    old = ns["_apply_rotary_pos_emb_bshd"](normalized * (1 + weight), freqs) * scale
    corrected = weight.clone()
    corrected[64:] = scale * (1 + weight[64:]) - 1
    new = ns["_apply_rotary_pos_emb_bshd"](normalized * (1 + corrected), freqs, mscale=scale)
    torch.testing.assert_close(new, old, rtol=1e-14, atol=1e-14)


@pytest.fixture
def tiny_export(tmp_path):
    scale = 1 + 0.1 * math.log(2)
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "head_dim": 8,
            "rope_parameters": {"partial_rotary_factor": 0.25, "attention_factor": scale},
            "layer_types": ["full_attention"],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors = {
        f"model.language_model.layers.0.self_attn.{kind}_norm.weight": torch.linspace(-0.05, 0.05, 8).to(
            torch.bfloat16
        )
        for kind in ("q", "k")
    }
    save_file(tensors, str(tmp_path / "model-00001.safetensors"))
    index = {"weight_map": dict.fromkeys(tensors, "model-00001.safetensors")}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    trained = {
        "mrope_yarn_enabled": True,
        "mrope_yarn_mscale_mode": "all_dimensions",
        "yarn_rotary_scaling_factor": 2.0,
        "yarn_mscale": 1.0,
        "yarn_mscale_all_dim": 0.0,
    }
    return tmp_path, trained, tensors


def test_compensation_backup_and_idempotency(tiny_export):
    root, trained, original = tiny_export
    result = compensate_legacy_mscale(root, trained)
    with safe_open(str(root / "model-00001.safetensors"), framework="pt") as f:
        for key, old in original.items():
            new = f.get_tensor(key)
            assert torch.equal(new[:2], old[:2])
            expected = ((old[2:].float() + 1) * result["scale"] - 1).to(old.dtype)
            assert torch.equal(new[2:], expected)
    before = (root / "model-00001.safetensors").read_bytes()
    assert compensate_legacy_mscale(root, trained)["already_applied"] is True
    assert (root / "model-00001.safetensors").read_bytes() == before
    journal = json.loads(Path(result["original_bytes_backup"]).read_text())
    for entry in journal["entries"]:
        old = torch.frombuffer(bytearray(base64.b64decode(entry["original_base64"])), dtype=torch.bfloat16)
        assert torch.equal(old, original[entry["key"]])


def test_interrupted_compensation_can_resume(tiny_export):
    root, trained, _ = tiny_export
    result = compensate_legacy_mscale(root, trained)
    path = Path(result["original_bytes_backup"])
    journal = json.loads(path.read_text())
    journal["status"] = "prepared"
    path.write_text(json.dumps(journal))
    entry = journal["entries"][0]
    with (root / entry["file"]).open("r+b") as stream:
        stream.seek(entry["offset"])
        stream.write(base64.b64decode(entry["original_base64"]))
    assert compensate_legacy_mscale(root, trained)["applied"] is True
    assert json.loads(path.read_text())["status"] == "complete"


def test_hardlinked_shard_is_rejected_before_writing(tiny_export):
    root, trained, _ = tiny_export
    weight = root / "model-00001.safetensors"
    before = weight.read_bytes()
    os.link(weight, root / "hardlink-backup.safetensors")
    with pytest.raises(ValueError, match="hard links"):
        compensate_legacy_mscale(root, trained)
    assert weight.read_bytes() == before


def test_single_file_compensation_is_equivalent_and_idempotent(tiny_export):
    root, trained, original = tiny_export
    (root / "model.safetensors.index.json").unlink()
    path = root / "model.safetensors"
    (root / "model-00001.safetensors").rename(path)
    report = compensate_legacy_mscale(root, trained)
    with safe_open(path, framework="pt") as weights:
        for key, old in original.items():
            corrected = weights.get_tensor(key)
            torch.testing.assert_close(corrected[:2], old[:2], rtol=0, atol=0)
            expected = ((old[2:].float() + 1) * report["scale"] - 1).to(old.dtype)
            torch.testing.assert_close(corrected[2:], expected, rtol=0, atol=0)
    before = path.read_bytes()
    assert compensate_legacy_mscale(root, trained)["already_applied"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("missing_weights", [False, True])
def test_legacy_finalizer_validates_weights_before_metadata(tiny_export, missing_weights):
    from llamafactory.model.model_utils.hf_export import finalize_hf_export

    root, trained, original = tiny_export
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    trained.update(max_sequence_length=16, yarn_original_max_position_embeddings=8)
    (checkpoint / "mca_config.json").write_text(json.dumps(trained))
    (root / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}))
    (root / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"content": "<eos>", "id": 2}]}))
    config = json.loads((root / "config.json").read_text())
    config["text_config"]["rope_parameters"]["attention_factor"] = 1.0
    (root / "config.json").write_text(json.dumps(config))
    before = (root / "config.json").read_bytes()
    if missing_weights:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        with pytest.raises(ValueError, match="Missing Q/K norm"):
            finalize_hf_export(checkpoint, root)
        assert (root / "config.json").read_bytes() == before
        assert not (root / "generation_config.json").exists()
    else:
        report = finalize_hf_export(checkpoint, root)
        assert report["weight_files_modified"]
        scale = report["legacy_mscale_compensation"]["scale"]
        assert (
            json.loads((root / "config.json").read_text())["text_config"]["rope_parameters"]["attention_factor"]
            == scale
        )
        assert (root / "export_config_backup/config.json").read_bytes() == before
        assert not finalize_hf_export(checkpoint, root)["weight_files_modified"]
    with safe_open(root / "model-00001.safetensors", framework="pt") as weights:
        for key, old in original.items():
            expected = old.clone()
            if not missing_weights:
                expected[2:] = ((old[2:].float() + 1) * scale - 1).to(old.dtype)
            torch.testing.assert_close(weights.get_tensor(key), expected, rtol=0, atol=0)

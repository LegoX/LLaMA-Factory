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

import json

import pytest

from llamafactory.model.model_utils.hf_export import finalize_hf_export


@pytest.fixture
def export_files(tmp_path):
    checkpoint, output, base = [tmp_path / name for name in ("ckpt", "hf", "base")]
    for directory in (checkpoint, output, base):
        directory.mkdir()
    trained = {
        "name_or_path": str(base),
        "mrope_yarn_enabled": True,
        "max_sequence_length": 524288,
        "yarn_rotary_scaling_factor": 2.0,
        "yarn_original_max_position_embeddings": 262144,
        "yarn_beta_fast": 32.0,
        "yarn_beta_slow": 1.0,
        "yarn_mscale": 1.0,
        "yarn_mscale_all_dim": 0.0,
        "yarn_correction_range_round_to_int": True,
    }
    rope = {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    }
    config = {"text_config": {"rope_parameters": rope, "eos_token_id": 43, "max_position_embeddings": 524288}}
    for directory, name, value in [
        (checkpoint, "mca_config.json", trained),
        (output, "config.json", config),
        (output, "tokenizer_config.json", {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}),
        (
            output,
            "tokenizer.json",
            {"added_tokens": [{"content": "<|im_end|>", "id": 42}, {"content": "<|endoftext|>", "id": 43}]},
        ),
        (base, "generation_config.json", {"eos_token_id": [42, 43], "temperature": 0.9, "do_sample": True}),
    ]:
        (directory / name).write_text(json.dumps(value))
    (output / "model.safetensors").write_bytes(b"weight bytes must not change")
    return checkpoint, output, base, config


def test_restores_trained_rope_and_generation_without_changing_weights(export_files):
    checkpoint, output, _, original = export_files
    before = (output / "config.json").read_bytes()
    result = finalize_hf_export(checkpoint, output)
    config = json.loads((output / "config.json").read_text())
    rope = config["text_config"]["rope_parameters"]
    assert rope["rope_type"] == "yarn" and rope["factor"] == 2.0
    assert rope["original_max_position_embeddings"] == 262144
    assert rope["truncate"] is True
    assert rope["mrope_section"] == [11, 11, 10] and rope["partial_rotary_factor"] == 0.25
    assert rope["attention_factor"] == pytest.approx(1.0693147180559945)
    generation = json.loads((output / "generation_config.json").read_text())
    assert generation["eos_token_id"] == [42, 43] and generation["pad_token_id"] == 43
    assert generation["temperature"] == 0.9
    assert config["text_config"]["eos_token_id"] == original["text_config"]["eos_token_id"]
    assert (output / "export_config_backup/config.json").read_bytes() == before
    assert (output / "model.safetensors").read_bytes() == b"weight bytes must not change"
    again = finalize_hf_export(checkpoint, output)
    assert again["changed_files"] == []
    assert (output / "export_config_backup/config.json").read_bytes() == before
    assert result["weight_files_modified"] is False


def test_no_yarn_is_added_to_native_context_model(export_files):
    checkpoint, output, _, original = export_files
    path = checkpoint / "mca_config.json"
    data = json.loads(path.read_text())
    data["mrope_yarn_enabled"] = False
    path.write_text(json.dumps(data))
    finalize_hf_export(checkpoint, output)
    assert json.loads((output / "config.json").read_text()) == original


def test_invalid_metadata_does_not_modify_export(export_files):
    checkpoint, output, _, _ = export_files
    before = (output / "config.json").read_bytes()
    path = checkpoint / "mca_config.json"
    data = json.loads(path.read_text())
    data["yarn_rotary_scaling_factor"] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="YaRN"):
        finalize_hf_export(checkpoint, output)
    assert (output / "config.json").read_bytes() == before
    assert not (output / "generation_config.json").exists()


def test_checkpoint_generation_settings_take_precedence(export_files):
    checkpoint, output, _, _ = export_files
    (checkpoint / "generation_config.json").write_text(json.dumps({"eos_token_id": 43, "temperature": 0.3}))
    finalize_hf_export(checkpoint, output)
    generation = json.loads((output / "generation_config.json").read_text())
    assert generation["temperature"] == 0.3 and generation["eos_token_id"] == [42, 43]


@pytest.mark.parametrize("maximum", [0, -1, True])
def test_invalid_explicit_context_length_is_not_silently_ignored(export_files, maximum):
    checkpoint, output, _, _ = export_files
    before = (output / "config.json").read_bytes()
    with pytest.raises(ValueError, match="context length"):
        finalize_hf_export(checkpoint, output, model_max_length=maximum)
    assert (output / "config.json").read_bytes() == before


def test_slow_tokenizer_special_ids_from_decoder(export_files):
    checkpoint, output, _, _ = export_files
    (output / "tokenizer.json").unlink()
    path = output / "tokenizer_config.json"
    data = json.loads(path.read_text())
    data["added_tokens_decoder"] = {"42": {"content": "<|im_end|>"}, "43": {"content": "<|endoftext|>"}}
    path.write_text(json.dumps(data))
    result = finalize_hf_export(checkpoint, output)
    assert result["tokenizer_eos_token_id"] == 42


def test_unsupported_legacy_export_does_not_rewrite_metadata(export_files):
    checkpoint, output, _, _ = export_files
    path = checkpoint / "mca_config.json"
    trained = json.loads(path.read_text())
    trained["mrope_yarn_mscale_mode"] = "all_dimensions"
    path.write_text(json.dumps(trained))
    before = (output / "config.json").read_bytes()
    with pytest.raises(ValueError, match="only verified"):
        finalize_hf_export(checkpoint, output)
    assert (output / "config.json").read_bytes() == before
    assert not (output / "generation_config.json").exists()
    assert (output / "model.safetensors").read_bytes() == b"weight bytes must not change"

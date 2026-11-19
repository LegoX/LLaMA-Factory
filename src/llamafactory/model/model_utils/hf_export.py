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

"""Finalize HF export metadata using the trained MCA configuration.

YaRN handling follows LegoX/LLaMA-Factory commit a8ec98361ab3,
scripts/megatron_merge.py::export_training_checkpoint. This also preserves
generation settings, which are not saved by MCoreAdapter's weight converter.
This file can run directly without CUDA or MCoreAdapter. Explicitly tagged
legacy checkpoints additionally compensate Q/K norms using CPU torch.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, data):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finalize_hf_export(checkpoint_path, output_path, base_model_path=None, model_max_length=None):
    """Restore metadata and, for tagged legacy checkpoints, reparameterize Q/K norms."""
    checkpoint, output = Path(checkpoint_path).resolve(), Path(output_path).resolve()
    if checkpoint == output:
        raise ValueError("HF output must differ from the native checkpoint")
    trained = _read(checkpoint / "mca_config.json")
    config_path = output / "config.json"
    config = _read(config_path)
    original = _read(config_path)
    text = config.get("text_config", config)
    yarn = bool(trained.get("mrope_yarn_enabled"))
    maximum = (
        model_max_length if model_max_length is not None else (trained.get("max_sequence_length") if yarn else None)
    )
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("Invalid exported context length")
        config["model_max_length"] = maximum
        if yarn:
            text["max_position_embeddings"] = maximum
            config["max_position_embeddings"] = maximum
    if yarn:
        if not maximum:
            raise ValueError("YaRN checkpoint has no max_sequence_length")
        factor = float(trained["yarn_rotary_scaling_factor"])
        original_max = int(trained["yarn_original_max_position_embeddings"])
        if factor <= 0 or not math.isfinite(factor) or original_max <= 0:
            raise ValueError("Invalid checkpoint YaRN scaling parameters")
        rope = dict(text.get("rope_parameters") or text.get("rope_scaling") or {})
        rope.update(
            rope_type="yarn",
            factor=factor,
            original_max_position_embeddings=original_max,
            beta_fast=float(trained.get("yarn_beta_fast", 32.0)),
            beta_slow=float(trained.get("yarn_beta_slow", 1.0)),
            mscale=float(trained.get("yarn_mscale", 1.0)),
            mscale_all_dim=float(trained.get("yarn_mscale_all_dim", 0.0)),
            truncate=bool(trained.get("yarn_correction_range_round_to_int", True)),
        )

        # Explicitly preserve the amplitude used by Megatron, including mscale_all_dim=0.
        def concentration(mscale):
            return 1.0 if factor <= 1 else 1.0 + 0.1 * mscale * math.log(factor)

        if any(not math.isfinite(rope[key]) or rope[key] < 0 for key in ("mscale", "mscale_all_dim")):
            raise ValueError("Invalid checkpoint YaRN amplitude")
        if any(not math.isfinite(rope[key]) or rope[key] <= 0 for key in ("beta_fast", "beta_slow")):
            raise ValueError("Invalid checkpoint YaRN correction range")
        rope["attention_factor"] = concentration(rope["mscale"]) / concentration(rope["mscale_all_dim"])
        text["rope_parameters"] = rope
        if text.get("rope_scaling") is not None:
            text["rope_scaling"] = dict(rope)

    base = Path(base_model_path or trained.get("name_or_path") or "")
    candidates = [checkpoint / "generation_config.json", output / "generation_config.json"]
    if base_model_path or trained.get("name_or_path"):
        candidates.append(base / "generation_config.json")
    source = next((p for p in candidates if p.is_file()), None)
    generation = _read(source) if source else {}
    tokenizer = _read(output / "tokenizer_config.json")
    vocab = _read(output / "tokenizer.json") if (output / "tokenizer.json").is_file() else {}
    special_ids = {t["content"]: t["id"] for t in vocab.get("added_tokens", [])}
    special_ids.update({t["content"]: int(idx) for idx, t in tokenizer.get("added_tokens_decoder", {}).items()})

    def token_id(field):
        token = tokenizer.get(field)
        if isinstance(token, dict):
            token = token.get("content")
        if token is None:
            return None
        value = special_ids.get(token)
        if value is None and isinstance(vocab.get("model", {}).get("vocab"), dict):
            value = vocab["model"]["vocab"].get(token)
        if value is None:
            # SentencePiece/slow tokenizers may keep the vocabulary in a binary file.
            from transformers import AutoTokenizer

            loaded = AutoTokenizer.from_pretrained(output, local_files_only=True)
            value = getattr(loaded, field + "_id", None)
        if value is None:
            raise ValueError(f"Cannot resolve {field}: {token!r}")
        return value

    eos = token_id("eos_token")
    if eos is None:
        raise ValueError("Exported tokenizer has no EOS")
    eos_ids = [eos]
    for values in (generation.get("eos_token_id"), text.get("eos_token_id"), config.get("eos_token_id")):
        if values is None:
            continue
        for value in values if isinstance(values, list) else [values]:
            if not isinstance(value, int) or value < 0:
                raise ValueError("Invalid EOS token ID")
            if value not in eos_ids:
                eos_ids.append(value)
    generation["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    for field in ("pad_token", "bos_token"):
        value = token_id(field)
        if value is not None:
            generation[field + "_id"] = value
    generation.pop("_from_model_config", None)

    # Reject invalid metadata or unsupported weights before changing export metadata.
    json.dumps(config, allow_nan=False)
    json.dumps(generation, allow_nan=False)
    compatibility = {"applied": False}
    if trained.get("mrope_yarn_mscale_mode") == "all_dimensions":
        if __package__:
            from .legacy_mscale import compensate_legacy_mscale
        else:
            from legacy_mscale import compensate_legacy_mscale
        compatibility = compensate_legacy_mscale(output, trained, config=config)

    backup = output / "export_config_backup"
    changed = []
    for path, payload in [(config_path, config), (output / "generation_config.json", generation)]:
        if path.is_file() and _read(path) == payload:
            continue
        if path.is_file():
            backup.mkdir(exist_ok=True)
            if not (backup / path.name).exists():
                shutil.copy2(path, backup / path.name)
        _write(path, payload)
        changed.append(path.name)
    report = {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "changed_files": changed,
        "yarn_enabled": yarn,
        "max_position_embeddings": text.get("max_position_embeddings"),
        "rope_parameters": text.get("rope_parameters"),
        "generation_source": str(source) if source else "model/tokenizer metadata",
        "eos_token_id": generation["eos_token_id"],
        "tokenizer_eos_token_id": eos,
        "weight_files_modified": compatibility.get("applied", False)
        and not compatibility.get("already_applied", False),
        "legacy_mscale_compensation": compatibility,
        "chat_template_modified": False,
        "previous_rope_parameters": original.get("text_config", original).get("rope_parameters"),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "reference": "https://github.com/LegoX/LLaMA-Factory/commit/a8ec98361ab3",
    }
    _write(output / "hf_export_metadata_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--model-max-length", type=int)
    args = parser.parse_args()
    print(
        json.dumps(finalize_hf_export(args.checkpoint, args.output, args.base_model, args.model_max_length), indent=2)
    )


if __name__ == "__main__":
    main()

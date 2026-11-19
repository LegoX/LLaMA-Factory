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

"""Reparameterize legacy Qwen3.5/3.8 all-dimension YaRN scaling for standard HF.

HF RMSNorm uses (1 + weight). On nonrotary coordinates only, replace weight
with mscale * (1 + weight) - 1. This moves the legacy extra Q/K scale before
RoPE, preserving its real-valued function without custom inference code.
Only explicitly tagged legacy checkpoints use this path. Original/new bytes
are journaled before any writes, allowing safe resumption and exact restoration.
"""

import base64
import fcntl
import json
import math
import os
import struct
import tempfile
from pathlib import Path


def _save(path, data):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compensate_legacy_mscale(output_path, trained, *, config=None):
    """Patch only Q/K norm payloads in existing safetensors, once per export."""
    if not trained.get("mrope_yarn_enabled") or trained.get("mrope_yarn_mscale_mode") != "all_dimensions":
        return {"applied": False, "reason": "checkpoint does not explicitly request legacy compatibility"}
    import torch

    root = Path(output_path)
    if config is None:
        config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Legacy norm compensation is only verified for Qwen3.5/3.8")
    text = config["text_config"]
    head_dim = text["head_dim"]
    rotary_dim = int(head_dim * text["rope_parameters"]["partial_rotary_factor"])
    factor = float(trained["yarn_rotary_scaling_factor"])

    def concentration(value):
        return 1.0 if factor <= 1 else 1.0 + 0.1 * value * math.log(factor)

    scale = concentration(trained.get("yarn_mscale", 1.0)) / concentration(trained.get("yarn_mscale_all_dim", 0.0))
    if not math.isfinite(scale) or scale <= 0 or not 0 < rotary_dim <= head_dim:
        raise ValueError("Invalid legacy compensation parameters")
    if not math.isclose(text["rope_parameters"]["attention_factor"], scale, rel_tol=1e-12):
        raise ValueError("Exported YaRN amplitude differs from the trained legacy amplitude")
    backup = root / "export_config_backup"
    backup.mkdir(exist_ok=True)
    journal_path = backup / "legacy_mscale_compensation.json"
    with (backup / "legacy_mscale.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            if journal["scale"] != scale or journal["rotary_dim"] != rotary_dim:
                raise ValueError("Existing compensation journal belongs to a different configuration")
        else:
            index_path = root / "model.safetensors.index.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text())["weight_map"]
            else:
                from safetensors import safe_open

                with safe_open(root / "model.safetensors", framework="pt") as weights:
                    index = dict.fromkeys(weights.keys(), "model.safetensors")
            keys = [
                f"model.language_model.layers.{i}.self_attn.{kind}_norm.weight"
                for i, kind_type in enumerate(text["layer_types"])
                if kind_type == "full_attention"
                for kind in ("q", "k")
            ]
            if not keys or any(key not in index for key in keys):
                raise ValueError("Missing Q/K norm weights for legacy compensation")
            entries = []
            dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
            for key in keys:
                filename = index[key]
                if Path(filename).name != filename:
                    raise ValueError("Unexpected shard path")
                if (root / filename).stat().st_nlink != 1:
                    raise ValueError("Break shard hard links before modifying Q/K norms: " + filename)
                with (root / filename).open("rb") as stream:
                    length = struct.unpack("<Q", stream.read(8))[0]
                    header = json.loads(stream.read(length))
                    item = header[key]
                    if item["shape"] != [head_dim] or item["dtype"] not in dtypes:
                        raise ValueError("Unexpected Q/K norm shape or dtype")
                    start, end = item["data_offsets"]
                    offset = 8 + length + start
                    stream.seek(offset)
                    raw = stream.read(end - start)
                old = torch.frombuffer(bytearray(raw), dtype=dtypes[item["dtype"]]).clone()
                if old.numel() != head_dim or not bool(torch.isfinite(old).all()):
                    raise ValueError("Invalid Q/K norm tensor payload")
                new = old.clone()
                new[rotary_dim:] = ((old[rotary_dim:].float() + 1.0) * scale - 1.0).to(old.dtype)
                new_raw = new.view(torch.uint8).numpy().tobytes()
                assert len(raw) == len(new_raw)
                entries.append(
                    {
                        "key": key,
                        "file": filename,
                        "offset": offset,
                        "original_base64": base64.b64encode(raw).decode(),
                        "compensated_base64": base64.b64encode(new_raw).decode(),
                    }
                )
            journal = {
                "schema": 1,
                "status": "prepared",
                "scale": scale,
                "head_dim": head_dim,
                "rotary_dim": rotary_dim,
                "entries": entries,
                "formula": "nonrotary weight = mscale * (1 + original weight) - 1",
            }
            _save(journal_path, journal)
        # Refuse unrelated edits; support recovery after interruption between tensor writes.
        already_complete = journal["status"] == "complete"
        for entry in journal["entries"]:
            original = base64.b64decode(entry["original_base64"])
            corrected = base64.b64decode(entry["compensated_base64"])
            with (root / entry["file"]).open("rb") as stream:
                stream.seek(entry["offset"])
                actual = stream.read(len(original))
            allowed = (corrected,) if already_complete else (original, corrected)
            if actual not in allowed:
                raise ValueError("Weight payload changed outside the compensation transaction: " + entry["key"])
        for entry in journal["entries"]:
            corrected = base64.b64decode(entry["compensated_base64"])
            with (root / entry["file"]).open("r+b") as stream:
                stream.seek(entry["offset"])
                if stream.read(len(corrected)) != corrected:
                    stream.seek(entry["offset"])
                    stream.write(corrected)
                    stream.flush()
                    os.fsync(stream.fileno())
        journal["status"] = "complete"
        _save(journal_path, journal)
    return {
        "applied": True,
        "already_applied": already_complete,
        "scale": scale,
        "tensors": len(journal["entries"]),
        "rotary_dim": rotary_dim,
        "nonrotary_dim": head_dim - rotary_dim,
        "original_bytes_backup": str(journal_path),
        "precision": "real-valued equivalence; BF16 retains normal quantization/rounding differences",
    }

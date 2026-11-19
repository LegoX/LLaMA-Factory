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

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import TrainingArguments

from llamafactory import launcher
from llamafactory.hparams.parser import read_args
from llamafactory.model.model_utils.megatron_export import convert_mca_to_hf


ROOT = Path(__file__).resolve().parents[2]


def preflight():
    spec = importlib.util.spec_from_file_location("preflight", ROOT / "examples/megatron/512k/preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "cudnn,flash,fused,torch_version,accepted",
    [
        ("9.26.0.51", None, None, "2.10.0+cu128", True),
        ("9.26.0.51", "1", "0", "2.10.0+cu128", True),
        ("9.10.2.21", "1", "0", "2.10.0+cu128", True),
        ("9.10.2.21", None, None, "2.10.0+cu128", False),
        ("9.10.2.21", "1", None, "2.10.0+cu128", False),
        ("9.10.2.21", "1", "1", "2.10.0+cu128", False),
        ("9.10.2.21", None, "0", "2.10.0+cu128", False),
        ("9.10.2.21", "0", "0", "2.10.0+cu128", False),
        ("9.11.0.98", "1", "0", "2.10.0+cu128", False),
        ("9.10.2.21", "1", "0", "2.9.0+cu128", False),
    ],
)
def test_preflight_versions_allow_torch_cudnn_only_with_explicit_flash(
    monkeypatch, cudnn, flash, fused, torch_version, accepted
):
    versions = {
        "torch": torch_version,
        "transformers": "5.6.0",
        "megatron-core": "0.18.2",
        "transformer-engine": "2.18.0",
        "flash-attn": "2.8.3.post1",
        "flash-linear-attention": "0.5.1",
        "fla-core": "0.5.1",
        "nvidia-cudnn-cu12": cudnn,
    }
    for name, value in (("NVTE_FLASH_ATTN", flash), ("NVTE_FUSED_ATTN", fused)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    module = preflight()
    monkeypatch.setattr(module.importlib.metadata, "version", versions.__getitem__)
    if accepted:
        module.validate_versions()
    else:
        with pytest.raises(ValueError, match="expected"):
            module.validate_versions()


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_config_resolution_matches_launcher_preflight_and_worker(monkeypatch, tmp_path, suffix):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config = tmp_path / ("training config" + suffix)
    values = dict(seed=42, output_dir="/tmp/run ${seed}", do_train=True, bf16=False)
    # JSON is also valid YAML; exercise the same interpolations for all accepted suffixes.
    config.write_text(json.dumps(values))
    overrides = ["seed=7", "bf16=true"]
    monkeypatch.setattr(sys, "argv", ["train", str(config), *overrides])
    expected = dict(values, seed=7, output_dir="/tmp/run 7", bf16=True)
    assert launcher._read_training_config() == read_args() == expected
    module = preflight()
    seen = []
    monkeypatch.setattr(module, "validate_topology", seen.append)
    monkeypatch.setattr(sys, "argv", ["preflight", "--config", str(config), "--config-only", "--", *overrides])
    module.main()
    assert seen == [expected]


def test_preflight_hides_gpus_before_transformers_import(tmp_path):
    config = tmp_path / "training.yaml"
    config.write_text("{}")
    # Use a fresh process: pytest has already imported Transformers in this one.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """\
import builtins
import os
import runpy
import sys

original_import = builtins.__import__
checked = []
def check_import(name, *args, **kwargs):
    if name == "transformers":
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Transformers imported before preflight hid GPUs"
        checked.append(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = check_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
assert checked, "Expected preflight to import Transformers"
""",
            str(ROOT / "examples/megatron/512k/preflight.py"),
            "--config",
            str(config),
            "--config-only",
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="2,5", NNODES="1", NODE_RANK="0", NPROC_PER_NODE="2"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Config-only validation passed" in result.stdout


def test_config_reader_preserves_explicit_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--output_dir", "a b"])
    assert read_args() == ["--output_dir", "a b"]
    assert launcher._read_training_config() is None
    supplied = {"output_dir": "literal ${seed}"}
    assert read_args(supplied) is supplied


def test_topology_uses_data_parallel_size(monkeypatch, capsys):
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("NPROC_PER_NODE", "8")
    monkeypatch.setenv("MASTER_ADDR", "node0")
    monkeypatch.setenv("NODE_RANK", "1")
    cfg = dict(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
    )
    preflight().validate_topology(cfg)
    assert "DP=2 GBS=64" in capsys.readouterr().out
    cfg["pretrain_score"] = True
    with pytest.raises(ValueError, match="CP1/DP1"):
        preflight().validate_topology(cfg)


def test_topology_resolves_checkpoint_and_override_precedence(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("NPROC_PER_NODE", "8")
    monkeypatch.setenv("NODE_RANK", "0")
    (tmp_path / "mca_config.json").write_text(
        json.dumps(dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=4, context_parallel_size=1))
    )
    config = dict(model_name_or_path=str(tmp_path), pretrain_score=True)
    preflight().validate_topology(config)
    assert "TP=2 PP=4 CP=1 DP=1" in capsys.readouterr().out
    config.update(
        pipeline_model_parallel_size=2,
        additional_configs=dict(tensor_model_parallel_size=4, pipeline_model_parallel_size=8),
    )
    preflight().validate_topology(config)
    assert "TP=4 PP=2 CP=1 DP=1" in capsys.readouterr().out
    config["additional_configs"]["context_parallel_size"] = 2
    with pytest.raises(ValueError, match="divisible"):
        preflight().validate_topology(config)


def test_topology_uses_training_argument_defaults(monkeypatch, capsys):
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("NPROC_PER_NODE", "8")
    monkeypatch.setenv("NODE_RANK", "0")
    preflight().validate_topology({})
    defaults = TrainingArguments.__dataclass_fields__
    expected = defaults["per_device_train_batch_size"].default * defaults["gradient_accumulation_steps"].default * 8
    assert f"DP=8 GBS={expected}" in capsys.readouterr().out


@pytest.mark.parametrize("packing_key", ["neat_packing", "sequence_packing"])
def test_topology_checks_cp_alignment_for_both_packing_flags(monkeypatch, packing_key):
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("NPROC_PER_NODE", "8")
    monkeypatch.setenv("NODE_RANK", "0")
    config = {packing_key: True, "context_parallel_size": 2}
    monkeypatch.delenv("LF_PACK_SUBSEQ_ALIGN", raising=False)
    with pytest.raises(ValueError, match="LF_PACK_SUBSEQ_ALIGN"):
        preflight().validate_topology(config)
    monkeypatch.setenv("LF_PACK_SUBSEQ_ALIGN", "2")
    with pytest.raises(ValueError, match="LF_PACK_SUBSEQ_ALIGN"):
        preflight().validate_topology(config)
    monkeypatch.setenv("LF_PACK_SUBSEQ_ALIGN", "8")
    preflight().validate_topology(config)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("resume", [None, "checkpoint-1"])
def test_only_node_zero_checks_existing_output(monkeypatch, tmp_path, rank, resume):
    monkeypatch.setenv("NODE_RANK", str(rank))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "dataset_info.json").write_text(json.dumps({"train": {"file_name": "train.json"}}))
    (tmp_path / "train.json").write_text("[]")
    output = tmp_path / "output"
    config = dict(
        model_name_or_path=str(tmp_path),
        dataset_dir=str(tmp_path),
        dataset="train",
        output_dir=str(output),
        resume_from_checkpoint=resume,
    )
    module = preflight()
    module.validate_input_paths(config)
    output.mkdir()  # Node 0 starts training before node 1 reaches this check.
    if rank == 0 and not resume:
        with pytest.raises(ValueError, match="Choose a new output directory"):
            module.validate_input_paths(config)
    else:
        module.validate_input_paths(config)
    (tmp_path / "train.json").unlink()
    with pytest.raises(ValueError, match="Missing dataset file"):
        module.validate_input_paths(config)


@pytest.mark.skipif(sys.platform != "linux", reason="The MCA launcher requires a Linux environment and GNU realpath")
def test_launcher_dry_run_applies_overrides_and_resets_single_node(tmp_path):
    config = tmp_path / "training.yaml"
    config.write_text("tensor_model_parallel_size: 4\npipeline_model_parallel_size: 4\n")
    env = dict(
        os.environ, DRY_RUN="1", NNODES="1", NODE_RANK="9", NPROC_PER_NODE="8", MASTER_ADDR="stale", USE_MCA="1"
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "examples/megatron/512k/run.sh"),
            sys.prefix,
            str(config),
            "pipeline_model_parallel_size=2",
            "per_device_train_batch_size=1",
            "gradient_accumulation_steps=64",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DP=1 GBS=64" in result.stdout and "[dry-run]" in result.stdout
    assert not (tmp_path / "saves").exists()


@pytest.mark.parametrize("elastic", [False, True])
def test_torchrun_preserves_argument_boundaries(monkeypatch, tmp_path, elastic):
    from llamafactory.extras import misc

    config = tmp_path / "training config.yaml"
    config.write_text("use_mca: true\ndo_train: true\nsave_hf_model: true\n")
    training_args = [str(config), "default_system=Be precise", f"output_dir={tmp_path / 'model output'}"]
    monkeypatch.setattr(sys, "argv", ["llamafactory-cli", "train", *training_args])
    monkeypatch.setattr(misc, "get_device_count", lambda: 8)
    monkeypatch.setattr(misc, "find_available_port", lambda: 29500)
    for name, value in dict(
        NNODES="2", NODE_RANK="1", NPROC_PER_NODE="8", MASTER_ADDR="node0", MASTER_PORT="29500"
    ).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("USE_MCA", "1")
    monkeypatch.setenv("FORCE_TORCHRUN", "1")
    monkeypatch.delenv("RDZV_ID", raising=False)
    if elastic:
        for name, value in dict(RDZV_ID="job id", MIN_NNODES="1", MAX_NNODES="2", MAX_RESTARTS="3").items():
            monkeypatch.setenv(name, value)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(SystemExit) as result:
        launcher.launch()
    assert result.value.code == 0 and len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[-4:] == [launcher.__file__, *training_args]
    assert argv[:3] == ["torchrun", "--nproc-per-node", "8"]
    assert argv[argv.index("--nnodes") + 1] == ("1:2" if elastic else "2")
    if elastic:
        assert argv[argv.index("--rdzv-id") + 1] == "job id"
        assert argv[argv.index("--rdzv-endpoint") + 1] == "node0:29500"
        assert argv[argv.index("--max-restarts") + 1] == "3"
    else:
        assert argv[argv.index("--node-rank") + 1] == "1"
        assert argv[argv.index("--master-addr") + 1] == "node0"
        assert argv[argv.index("--master-port") + 1] == "29500"
    assert kwargs["check"] and kwargs.get("shell", False) is False


@pytest.mark.parametrize(
    "flags", [{"pretrain_score_only": True}, {"benchmark_skip_final_save": True}, {"save_hf_model": True}]
)
def test_no_export_when_no_native_final_save(monkeypatch, flags):
    monkeypatch.setenv("USE_MCA", "1")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected conversion"))
    launcher._export_after_training(dict(do_train=True, **flags), 0)


def test_export_runs_packaged_converter_once_with_clean_rank_env(monkeypatch):
    monkeypatch.setenv("USE_MCA", "1")
    monkeypatch.setenv("RANK", "12")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    launcher._export_after_training(dict(do_train=True), 1)
    assert not calls
    launcher._export_after_training(dict(do_train=True), 0)
    args, kwargs = calls[0]
    assert args[0][1:3] == ["-m", "llamafactory.model.model_utils.megatron_export"]
    assert "WORLD_SIZE" not in kwargs["env"] and "RANK" not in kwargs["env"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3" and kwargs["check"]


def test_conversion_rejects_in_place_and_nonempty_outputs_before_loading_cuda(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        convert_mca_to_hf(str(tmp_path), str(tmp_path))
    output = tmp_path.parent / (tmp_path.name + "-hf")
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError):
        convert_mca_to_hf(str(tmp_path), str(output))
    assert (output / "sentinel").read_text() == "preserve"

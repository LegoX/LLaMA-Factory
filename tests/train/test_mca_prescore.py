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

"""CPU checks for source identity, truncation, shifted losses and PP dispatch."""

import ast
import copy
import functools
import importlib.util
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset

from llamafactory.data.converter import SharegptDatasetConverter, _convert_with_score_source
from llamafactory.data.parser import DatasetAttr
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.hparams import DataArguments


ROOT = Path(__file__).resolve().parents[2]
# Load the scorer without importing mca/__init__, which requires the CUDA adapter.
_spec = importlib.util.spec_from_file_location(
    "llamafactory.train.mca.scoring", ROOT / "src/llamafactory/train/mca/scoring.py"
)
_scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scoring)
aggregate_sample = _scoring.aggregate_sample
score_training_dataset = _scoring.score_training_dataset
validate_scoring_config = _scoring.validate_scoring_config


def collate(features):
    length = max(len(row["input_ids"]) for row in features) + 2
    return {
        key: torch.tensor([row[key] + [pad] * (length - len(row[key])) for row in features])
        for key, pad in [("input_ids", 0), ("labels", -100), ("attention_mask", 0)]
    }


def wrapper():
    # The workflow imports the CUDA-only adapter; execute its production wrapper on CPU.
    path = ROOT / "src/llamafactory/train/mca/workflow.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_data_collator_wrapper"
    )
    ns = dict(functools=functools, deepcopy=copy.deepcopy, Any=Any, Sequence=Sequence, torch=torch, IGNORE_INDEX=-100)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[node.name](collate)


def processed(cutoff=100):
    args = DataArguments(pretrain_score=True, cutoff_len=cutoff)
    attr = DatasetAttr("file", "source.json")
    converter = SharegptDatasetConverter(attr, args)
    source = Dataset.from_list(
        [
            {
                "id": "original-42",
                "_score_source_index": 42,
                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}] * 3,
            }
        ]
    )
    aligned = source.map(
        functools.partial(_convert_with_score_source, converter=converter), remove_columns=source.column_names
    )
    mm = SimpleNamespace(
        process_messages=lambda messages, *args: messages, process_token_ids=lambda ids, labels, *args: (ids, labels)
    )
    template = SimpleNamespace(
        mm_plugin=mm,
        efficient_eos=False,
        preserve_thinking=True,
        encode_multiturn=lambda *args: [([1, 2], [3, 4]), ([5, 6], [7, 8, 9]), ([1], [2])],
    )
    processor = SupervisedDatasetProcessor(template, SimpleNamespace(eos_token_id=0), None, args)
    return aligned.map(processor.preprocess_dataset, batched=True, remove_columns=aligned.column_names)


def test_source_identity_turn_spans_and_truncation_survive_arrow():
    row = processed(cutoff=6)[0]
    metadata = json.loads(row["score_metadata"])
    assert metadata["source_row_index"] == 42 and metadata["original_id"] == "original-42"
    assert metadata["dataset"] == "source.json" and metadata["full_length"] == 11
    assert metadata["truncated"] and metadata["retained_length"] == 6
    assert [(t["start"], t["end"]) for t in metadata["turns"]] == [(2, 4), (5, 6), (None, None)]
    assert metadata["turns"][1]["target_truncated"] and metadata["turns"][1]["context_truncated"]
    before = copy.deepcopy(row)
    batch = wrapper()([row])
    assert row == before and "score_metadata" not in batch
    losses = torch.arange(batch["labels"].numel(), dtype=torch.float32).reshape_as(batch["labels"])
    result = aggregate_sample(losses, batch["labels"], metadata, 0)
    assert result["turns"][0]["loss"] == 1.5  # shifted token indices 1, 2
    assert result["turns"][1]["loss"] == 4.0  # shifted token index 4
    assert result["loss"] == pytest.approx(7 / 3) and result["num_loss_tokens"] == 3
    assert result["turns"][2]["loss"] is None and result["turns"][2]["status"] == "no_valid_tokens"


def test_cross_entropy_matches_direct_reference_and_length_weighting():
    row = processed()[0]
    metadata = json.loads(row["score_metadata"])
    batch = wrapper()([row])
    torch.manual_seed(3)
    logits = torch.randn(1, batch["labels"].shape[1], 12)
    labels = batch["labels"]
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)
    result = aggregate_sample(losses, labels, metadata, 7)
    expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten()).item()
    assert result["loss"] == pytest.approx(expected)
    assert result["num_loss_tokens"] == 6
    assert result["loss"] == pytest.approx(sum(t["loss"] * t["num_loss_tokens"] for t in result["turns"]) / 6)
    for turn in result["turns"]:
        assert turn["ppl"] == pytest.approx(math.exp(turn["loss"]))
    # Identical score under left padding, with all shifted spans offset together.
    length = metadata["retained_length"] - 1
    left_labels = F.pad(labels[:, :length], (2, 0), value=-100)
    left_losses = F.pad(losses[:, :length], (2, 0))
    assert aggregate_sample(left_losses, left_labels, metadata, 7, "left") == result


@pytest.mark.parametrize("value,status", [(float("nan"), "nonfinite_loss"), (1000.0, "ppl_overflow")])
def test_nonfinite_scores_are_valid_json(value, status):
    row = processed()[0]
    batch = wrapper()([row])
    result = aggregate_sample(
        torch.full_like(batch["labels"], value, dtype=torch.float32),
        batch["labels"],
        json.loads(row["score_metadata"]),
        0,
    )
    assert result["status"] == status and result["ppl"] is None
    json.dumps(result, allow_nan=False)


def test_unassigned_supervision_is_rejected():
    row = processed()[0]
    batch = wrapper()([row])
    batch["labels"][0, 0] = 2
    with pytest.raises(ValueError, match="no assistant turn"):
        aggregate_sample(torch.zeros_like(batch["labels"]), batch["labels"], json.loads(row["score_metadata"]), 0)


@pytest.mark.parametrize(
    "option",
    [
        "packing",
        "streaming",
        "train_on_prompt",
        "mask_history",
        "turn_mask",
        "trajectory_loss_weight",
        "drop_long_sequences",
    ],
)
def test_unsupported_data_modes_fail_early(option):
    with pytest.raises(ValueError, match="pretrain_score"):
        DataArguments(pretrain_score=True, **{option: True})


def test_parallel_config_guard(monkeypatch):
    args = SimpleNamespace(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_packing=False,
        variable_seq_lengths=True,
        resume_from_checkpoint=None,
    )
    monkeypatch.setenv("WORLD_SIZE", "16")
    validate_scoring_config(args, args, DataArguments(pretrain_score=True))
    args.variable_seq_lengths = False
    with pytest.raises(ValueError, match="variable_seq_lengths"):
        validate_scoring_config(args, args, DataArguments(pretrain_score=True))
    validate_scoring_config(args, args, DataArguments(pretrain_score=True, pretrain_score_microbatches=1))
    monkeypatch.setenv("WORLD_SIZE", "32")
    with pytest.raises(ValueError, match="CP1/DP1"):
        validate_scoring_config(args, args, DataArguments(pretrain_score=True))


@pytest.mark.parametrize("cp", [1, 2])
def test_scoring_validates_resolved_config_before_model_allocation(monkeypatch, cp):
    path = ROOT / "src/llamafactory/train/mca/workflow.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "run_sft")
    config = SimpleNamespace(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        context_parallel_size=cp,
        virtual_pipeline_model_parallel_size=None,
        variable_seq_lengths=True,
    )
    # Top-level None values must not override the configuration resolved from the checkpoint/overrides.
    args = SimpleNamespace(
        tensor_model_parallel_size=None,
        pipeline_model_parallel_size=None,
        context_parallel_size=None,
        sequence_packing=False,
        resume_from_checkpoint=None,
    )
    allocations = []

    class AllocationReached(Exception):
        pass

    def allocate(path, training_args, **kwargs):
        allocations.append(kwargs["config"])
        raise AllocationReached

    namespace = dict(
        __package__="llamafactory.train.mca",
        Optional=Optional,
        load_tokenizer=lambda *a: {"tokenizer": object()},
        get_template_and_fix_tokenizer=lambda *a: None,
        get_dataset=lambda *a, **kw: {},
        _check_model_support=lambda *a: None,
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a: config),
        AutoModel=SimpleNamespace(from_pretrained=allocate),
    )
    monkeypatch.setitem(sys.modules, "llamafactory.train.mca.scoring", _scoring)
    monkeypatch.setenv("WORLD_SIZE", "8")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(AllocationReached if cp == 1 else ValueError, match=None if cp == 1 else "CP1/DP1"):
        namespace["run_sft"](
            SimpleNamespace(model_name_or_path="checkpoint"),
            DataArguments(pretrain_score=True),
            args,
            SimpleNamespace(),
        )
    assert allocations == ([config] if cp == 1 else [])


@pytest.mark.parametrize("microbatches", [1, 2, 16])
def test_scoring_loop_uses_forward_only_and_restores_modes(tmp_path, monkeypatch, microbatches, megatron_distribution):
    full, short = processed()[0], processed(cutoff=6)[0]
    dataset = Dataset.from_list([full, short, full, short, full])
    models = [torch.nn.Linear(2, 2)]
    models[0].train()
    original_weights = models[0].weight.detach().clone()
    mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_data_parallel_world_size=lambda: 1,
        is_pipeline_last_stage=lambda: True,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 4,
        get_pipeline_model_parallel_world_size=lambda: 4,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(parallel_state=mpu))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 12)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    real_fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: real_fork(devices=[]))
    # Exercise the installed MCore dispatch contract, including non-final stages.
    path = Path(megatron_distribution.locate_file("megatron/core/pipeline_parallel/schedules.py"))
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_step_calc_loss"
    )
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    dispatch = namespace[node.name]
    monkeypatch.setitem(
        sys.modules, "megatron.core.transformer.multi_token_prediction", SimpleNamespace(MTPLossAutoScaler=object)
    )
    config = SimpleNamespace(calculate_per_token_loss=False, timers=None)
    calls = []

    def schedule(**kwargs):
        assert kwargs["forward_only"] and kwargs["micro_batch_size"] == 1
        assert not torch.is_grad_enabled() and not models[0].training
        items = list(kwargs["data_iterator"][0])
        assert kwargs["num_microbatches"] == len(items)
        assert kwargs["seq_length"] == max(batch["input_ids"].shape[1] for _, _, batch in items)
        assert len({batch["input_ids"].shape[1] for _, _, batch in items}) == min(len(items), 2)
        # Delayed callbacks catch late-bound metadata/index bugs across microbatches.
        deferred = []
        for index, metadata, batch in items:
            for last_stage in [False, True]:
                value = (
                    torch.ones_like(batch["labels"], dtype=torch.float32) * (index + 1)
                    if last_stage
                    else torch.randn(4, 1, 2)
                )
                outputs, callback = kwargs["forward_step_func"](
                    iter([(index, metadata, batch)]), lambda **inputs: value
                )
                deferred.append((last_stage, outputs, callback))
        for last_stage, outputs, callback in deferred:
            store = []
            actual, _ = dispatch(
                SimpleNamespace(),
                outputs,
                callback,
                config,
                None,
                False,
                len(items),
                store,
                cp_group_size=1,
                is_last_stage=last_stage,
            )
            if not last_stage:
                assert actual is outputs and store == []
        calls.append(len(items))

    trainer = SimpleNamespace(
        model=SimpleNamespace(
            get_models=lambda: models, config=SimpleNamespace(mtp_num_layers=0, variable_seq_lengths=True)
        ),
        train_dataset=dataset,
        processing_class=SimpleNamespace(padding_side="right"),
        args=SimpleNamespace(device="cpu", output_dir=str(tmp_path)),
        data_collator=wrapper(),
        _prepare_inputs=lambda batch: batch,
        _prepare_train_inputs=lambda iterator: dict(next(iterator)),
        forward_backward_func=schedule,
    )
    score_training_dataset(trainer, "initial-model", microbatches)
    rows = [json.loads(line) for line in (tmp_path / "pretrain_scores.jsonl").read_text().splitlines()]
    assert len(rows) == len(dataset) == 5
    assert calls == [min(microbatches, 5 - i) for i in range(0, 5, microbatches)]
    assert [row["sample_index"] for row in rows] == list(range(5))
    assert [row["loss"] for row in rows] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [row["num_loss_tokens"] for row in rows] == [6, 3, 6, 3, 6]
    assert rows[0]["source_row_index"] == 42 and rows[0]["num_loss_tokens"] == 6
    assert models[0].training
    torch.testing.assert_close(models[0].weight, original_weights)
    assert models[0].weight.grad is None
    assert not (tmp_path / "pretrain_scores.jsonl.incomplete").exists()
    manifest = json.loads((tmp_path / "pretrain_scores.meta.json").read_text())
    assert manifest["num_samples"] == 5 and manifest["model_name_or_path"] == "initial-model"
    assert manifest["scoring_microbatches"] == microbatches
    with pytest.raises(RuntimeError, match="already exists"):
        score_training_dataset(trainer, "initial-model", microbatches)
    assert models[0].training


def test_qwen36_real_tokenizer_preserves_reasoning_and_turn_boundaries():
    from transformers import AutoProcessor, AutoTokenizer

    from llamafactory.data import get_template_and_fix_tokenizer

    model_path = Path(os.getenv("QWEN36_TOKENIZER", "__unavailable__"))
    if not model_path.is_dir():
        pytest.skip("Local Qwen3.6 tokenizer is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    args = DataArguments(pretrain_score=True, template="qwen3_6", preserve_thinking=True, cutoff_len=4096)
    template = get_template_and_fix_tokenizer(tokenizer, args)
    messages = [
        {"role": "user", "content": "Compute 1+1."},
        {"role": "assistant", "content": "<think>\nAdd one and one.\n</think>\n\n2"},
        {"role": "user", "content": "And 2+2?"},
        {"role": "assistant", "content": "<think>\nAdd again.\n</think>\n\n4"},
    ]
    processor = SupervisedDatasetProcessor(
        template, tokenizer, AutoProcessor.from_pretrained(model_path, local_files_only=True), args
    )
    metadata = {}
    ids, labels, _, full_length = processor._encode_data_example(
        messages[:-1], messages[-1:], "Be precise.", None, [], [], [], metadata
    )
    expected_pairs = template.encode_multiturn(tokenizer, messages, "Be precise.")
    assert ids == [token for source, target in expected_pairs for token in source + target]
    assert full_length == len(ids) and len(metadata["turns"]) == 2
    for turn, (_, target) in zip(metadata["turns"], expected_pairs):
        assert labels[turn["start"] : turn["end"]] == target
        assert "Add" in tokenizer.decode(target)
        assert not turn["target_truncated"] and not turn["context_truncated"]
    assert sum(label != -100 for label in labels) == sum(len(target) for _, target in expected_pairs)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_scoring_microbatches(value):
    with pytest.raises(ValueError, match="positive integer"):
        DataArguments(pretrain_score_microbatches=value)

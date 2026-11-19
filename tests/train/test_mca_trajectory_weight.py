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

"""CPU regression tests on the production MCA/data functions; no model load."""

import ast
import copy
import functools
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from llamafactory.data.converter import SharegptDatasetConverter
from llamafactory.data.parser import DatasetAttr
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.hparams import DataArguments


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "src/mcore_adapter/trainer/trainer.py"


def collator_wrapper(collator, weighted=True):
    path = ROOT / "src/llamafactory/train/mca/workflow.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_data_collator_wrapper")
    ns = dict(functools=functools, deepcopy=copy.deepcopy, Any=Any, Sequence=Sequence, torch=torch, IGNORE_INDEX=-100)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)
    return ns[function.name](collator, trajectory_loss_weight=weighted)


def trainer_class():
    tree = ast.parse(ADAPTER.read_text())
    post = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_post_compute_loss")
    ns = dict(torch=torch, mpu=SimpleNamespace(get_context_parallel_group=lambda: None))
    exec(compile(ast.Module(body=[post], type_ignores=[]), str(ADAPTER), "exec"), ns)
    inner = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_inner_forward_step")
    ns["partial"] = functools.partial
    exec(
        compile(ast.Module(body=[inner], type_ignores=[]), str(ADAPTER), "exec"),
        ns := dict(ns, Iterator=Any, DistributedDataParallel=Any),
    )
    pre = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_pre_compute_loss")
    ns["IGNORE_INDEX"] = -100
    exec(compile(ast.Module(body=[pre], type_ignores=[]), str(ADAPTER), "exec"), ns)

    base = type(
        "Base",
        (),
        {
            "_pre_compute_loss": ns[pre.name],
            "_post_compute_loss": ns[post.name],
            "_inner_forward_step": ns[inner.name],
        },
    )
    path = ROOT / "src/llamafactory/train/mca/trainer.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    ns = dict(
        McaTrainer=base,
        torch=torch,
        os=os,
        time=time,
        json=json,
        IGNORE_INDEX=-100,
        Tensor=torch.Tensor,
        Any=Any,
        F=F,
        override=lambda f: f,
        PreTrainedTokenizerBase=type("TokenizerBase", (), {}),
        logger=SimpleNamespace(info=lambda *a: None),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    return ns[cls.name]


@pytest.mark.parametrize("bad", [None, True, False, "0.5", 0, -1, float("nan"), float("inf"), 2])
def test_converter_rejects_invalid_weight(bad):
    converter = SharegptDatasetConverter(
        DatasetAttr("file", "x", loss_weight="loss_weight"), DataArguments(trajectory_loss_weight=True)
    )
    row = dict(
        conversations=[dict(**{"from": "human", "value": "q"}), dict(**{"from": "gpt", "value": "a"})], loss_weight=bad
    )
    with pytest.raises(ValueError, match="loss_weight"):
        converter(row)


def test_registered_column_survives_arrow_map_and_processor_truncation():
    from datasets import Dataset

    attr = DatasetAttr("file", "weighted")
    attr.join(
        dict(
            formatting="sharegpt",
            columns=dict(messages="messages", loss_weight="loss_weight"),
            tags=dict(role_tag="role", content_tag="content", user_tag="user", assistant_tag="assistant"),
        )
    )
    data_args = DataArguments(trajectory_loss_weight=True, turn_mask=True, cutoff_len=5)
    converter = SharegptDatasetConverter(attr, data_args)
    row = dict(
        messages=[
            dict(role="user", content="q1"),
            dict(role="assistant", content="a1", turn_mask=True),
            dict(role="user", content="q2"),
            dict(role="assistant", content="a2"),
        ],
        loss_weight=0.5,
    )

    class MM:
        def process_messages(self, messages, *a):
            return messages

        def process_token_ids(self, ids, labels, *a):
            return ids, labels

    template = SimpleNamespace(
        mm_plugin=MM(),
        efficient_eos=False,
        preserve_thinking=True,
        encode_multiturn=lambda *a: [([10], [11, 12]), ([20], [21, 22])],
    )
    processor = SupervisedDatasetProcessor(template, SimpleNamespace(eos_token_id=2), None, data_args)
    aligned = Dataset.from_list([row]).map(converter, remove_columns=list(row))
    processed = aligned.map(processor.preprocess_dataset, batched=True, remove_columns=aligned.column_names)
    assert processed[0]["trajectory_loss_weight"] == 0.5
    assert processed[0]["labels"] == [-100, -100, -100, -100, 21]
    assert processed[0]["input_ids"] == [10, 11, 12, 20, 21]
    # Opt-out keeps the original schema/labels and ignores extra weight metadata.
    data_args.trajectory_loss_weight = False
    aligned_plain = converter(row)
    assert "_loss_weight" not in aligned_plain


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("weight", [0.25, 0.5, 1.0])
def test_microbatch_one_gradient_scales_without_denominator_cancellation(weight, per_token):
    obj = trainer_class()()
    obj.args = SimpleNamespace(calculate_per_token_loss=per_token)
    obj.model = SimpleNamespace(config=SimpleNamespace(context_parallel_size=1))
    labels = torch.tensor([[-100, 1, 2, -100, 0]])
    weights = torch.full(labels.shape, weight)
    obj._prepare_train_inputs = lambda _: dict(labels=labels, loss_weights=weights)
    logits = torch.tensor([[[1.0, 2.0, 3.0]] * 5], requires_grad=True)

    def model(**inputs):
        assert "loss_weights" not in inputs
        torch.testing.assert_close(inputs["loss_mask"], (labels != -100).float())
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)

    weighted, mask = obj._pre_compute_loss(None, model)
    loss, tokens, _ = obj._post_compute_loss(mask, weighted)
    assert tokens.item() == 3
    (loss / tokens).backward()
    actual = logits.grad.clone()
    logits.grad = None
    F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
    torch.testing.assert_close(actual, logits.grad * weight)
    assert actual[0, 0].abs().sum() == actual[0, 3].abs().sum() == 0


def test_mixed_weight_batch_shift_padding_and_no_inplace_changes():
    def collate(features):
        n = max(len(f["input_ids"]) for f in features)
        return {
            k: torch.tensor([f[k] + [pad] * (n - len(f[k])) for f in features])
            for k, pad in [("input_ids", 0), ("labels", -100), ("attention_mask", 0)]
        }

    rows = [
        dict(input_ids=[1, 2, 3, 4], labels=[-100, 2, -100, 4], attention_mask=[1] * 4, trajectory_loss_weight=0.5),
        dict(input_ids=[1, 2, 3], labels=[-100, 2, 3], attention_mask=[1] * 3, trajectory_loss_weight=1.0),
    ]
    before = copy.deepcopy(rows)
    out = collator_wrapper(collate)(rows)
    assert rows == before
    torch.testing.assert_close(out["loss_weights"], torch.tensor([[0.5, 0.0, 0.5], [1.0, 1.0, 0.0]]))
    assert out["labels"].tolist() == [[2, -100, 4], [2, 3, -100]]


@pytest.mark.parametrize("side", ["left", "right"])
def test_fixed_sequence_padding_preserves_weight_alignment(side):
    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        padding_side = side

        def pad(self, inputs, max_length, **kwargs):
            result = Batch()
            for k, rows in inputs.items():
                pad = -100 if k == "labels" else 0
                result[k] = torch.tensor(
                    [
                        ([pad] * (max_length - len(r)) + r if side == "left" else r + [pad] * (max_length - len(r)))
                        for r in rows
                    ]
                )
            return result

    obj = trainer_class()()
    obj._language_input_names = ["input_ids", "attention_mask", "labels", "position_ids"]
    obj.processing_class = Tokenizer()
    obj.args = SimpleNamespace(device="cpu")
    out = obj._pad_batched_inputs(
        dict(
            input_ids=torch.tensor([[1, 2]]), labels=torch.tensor([[-100, 2]]), loss_weights=torch.tensor([[0.0, 0.5]])
        ),
        4,
    )
    expected = [[0.0, 0.0, 0.0, 0.5]] if side == "left" else [[0.0, 0.5, 0.0, 0.0]]
    assert out["loss_weights"].tolist() == expected


def test_packing_and_missing_explicit_weight_fail_closed():
    with pytest.raises(ValueError, match="unpacked"):
        DataArguments(trajectory_loss_weight=True, packing=True)
    converter = SharegptDatasetConverter(DatasetAttr("file", "x"), DataArguments(trajectory_loss_weight=True))
    with pytest.raises(ValueError, match="explicit"):
        converter(dict(conversations=[{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]))


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp", [2, 4])
def test_weighted_cp_reduction_preserves_unweighted_denominator(cp, per_token):
    obj = trainer_class()()
    obj.model = SimpleNamespace(config=SimpleNamespace(context_parallel_size=cp))
    obj.args = SimpleNamespace(calculate_per_token_loss=per_token)
    x = torch.arange(1.0, 9.0)
    mask = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    weights = torch.tensor([1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5])
    reference = torch.tensor(0.3, requires_grad=True)
    ((reference * x).square() * weights * mask).sum().div(mask.sum()).backward()
    grads = []
    for rank in range(cp):
        local_x, local_mask, local_weights = [t.chunk(cp)[rank] for t in (x, mask, weights)]
        param = torch.tensor(0.3, requires_grad=True)
        # The production model's CP slicer treats loss_weights as a sequence
        # tensor. Feed the resulting local tensors into the real loss path.
        labels = torch.where(local_mask.bool(), 1, -100).reshape(1, -1)
        obj._prepare_train_inputs = lambda _: dict(labels=labels, loss_weights=local_weights.reshape(1, -1))
        losses, binary = obj._pre_compute_loss(None, lambda **kw: (param * local_x).square().reshape(1, -1))
        total = ((param.detach() * x).square() * weights * mask).sum()
        local = ((param.detach() * local_x).square() * local_weights * local_mask).sum()
        remote = torch.stack([total - local, mask.sum() - local_mask.sum()])

        def reduce(tensor, **kwargs):
            with torch.no_grad():
                tensor.add_(remote)

        with patch.object(torch.distributed, "all_reduce", side_effect=reduce):
            loss, tokens, _ = obj._post_compute_loss(binary, losses)
        assert tokens == mask.sum()
        (loss / (tokens * cp if per_token else tokens)).backward()
        grads.append(param.grad)
    actual = torch.stack(grads).sum() if per_token else torch.stack(grads).mean()
    torch.testing.assert_close(actual, reference.grad)


def test_unsupported_dataset_formats_fail_before_dropping_weights():
    from llamafactory.data.converter import get_dataset_converter

    with pytest.raises(ValueError, match="ShareGPT"):
        get_dataset_converter("alpaca", DatasetAttr("file", "x"), DataArguments(trajectory_loss_weight=True))


@pytest.fixture
def pipeline_loss_dispatch(megatron_distribution):
    # Execute the installed scheduler's real last/non-last-stage dispatch on
    # CPU. Only its unused MTP scaler import is substituted; no CUDA stack.
    path = Path(megatron_distribution.locate_file("megatron/core/pipeline_parallel/schedules.py"))
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_step_calc_loss"
    )
    ns = dict(torch=torch)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)
    return ns[function.name]


@pytest.mark.parametrize("per_token", [False, True])
def test_trajectory_weight_scales_ce_but_not_moe_auxiliary_loss(
    per_token, pipeline_loss_dispatch, megatron_distribution
):
    path = Path(megatron_distribution.locate_file("megatron/core/transformer/moe/moe_utils.py"))
    scaler = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MoEAuxLossAutoScaler"
    )
    ns = dict(torch=torch, Optional=Optional, Tuple=tuple)
    exec(compile(ast.Module(body=[scaler], type_ignores=[]), str(path), "exec"), ns)
    scaler = ns[scaler.name]
    config = SimpleNamespace(
        context_parallel_size=1,
        calculate_per_token_loss=per_token,
        timers=None,
        num_moe_experts=2,
        grad_scale_func=None,
    )
    obj = trainer_class()()
    obj.args = config
    obj.model = SimpleNamespace(config=config)
    labels = torch.tensor([[1, 2]])
    reference = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]], requires_grad=True)
    ce_grad = torch.autograd.grad(F.cross_entropy(reference.flatten(0, 1), labels.flatten()), reference)[0]
    aux_grad = torch.autograd.grad(reference.square().mean() * 0.1, reference)[0]
    if per_token:
        aux_grad /= labels.numel()

    for weight in (1.0, 0.5):
        logits = reference.detach().clone().requires_grad_()
        output = scaler.apply(logits, logits.square().mean() * 0.1)
        losses = F.cross_entropy(output.flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)
        callback = functools.partial(
            obj._post_compute_loss, (torch.ones_like(losses), torch.full_like(losses, weight))
        )
        with (
            patch.dict(
                pipeline_loss_dispatch.__globals__,
                MoEAuxLossAutoScaler=scaler,
                get_tensor_device=lambda tensor: tensor.device,
            ),
            patch.dict(
                sys.modules,
                {"megatron.core.transformer.multi_token_prediction": SimpleNamespace(MTPLossAutoScaler=object)},
            ),
        ):
            loss, tokens = pipeline_loss_dispatch(
                SimpleNamespace(), losses, callback, config, None, False, 1, [], cp_group_size=1, is_last_stage=True
            )
        loss.backward()
        if per_token:
            # finalize_model_grads divides the accumulated parameter gradients by the token count.
            logits.grad /= tokens
        torch.testing.assert_close(logits.grad, weight * ce_grad + aux_grad)
        if weight == 0.5:
            assert not torch.allclose(logits.grad, weight * (ce_grad + aux_grad))


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp", [1, 2, 4])
@pytest.mark.parametrize("all_masked", [False, True])
def test_masked_microbatch_accumulation_matches_reference(cp, per_token, all_masked, pipeline_loss_dispatch):
    dispatch = pipeline_loss_dispatch
    obj = trainer_class()()
    obj.args = SimpleNamespace(calculate_per_token_loss=per_token)
    obj.model = SimpleNamespace(config=SimpleNamespace(context_parallel_size=cp))
    config = SimpleNamespace(calculate_per_token_loss=per_token, timers=None)
    x = torch.arange(1.0, 9.0)
    masks = [torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), torch.zeros(8)]
    if all_masked:
        masks[0].zero_()
    reference = torch.tensor(0.3, requires_grad=True)
    numerators = [((reference * x).square() * mask * 0.5).sum() for mask in masks]
    if per_token:
        expected = sum(numerators) / sum(mask.sum() for mask in masks).clamp_min(1)
    else:
        expected = sum(loss / mask.sum().clamp_min(1) for loss, mask in zip(numerators, masks)) / len(masks)
    expected.backward()

    grads = []
    stub = SimpleNamespace(MTPLossAutoScaler=object)
    for rank in range(cp):
        param = torch.tensor(0.3, requires_grad=True)
        local_x = x.chunk(cp)[rank]
        losses, counts = [], []
        for mask in masks:
            local_mask = mask.chunk(cp)[rank]
            total = ((param.detach() * x).square() * mask * 0.5).sum()
            local = ((param.detach() * local_x).square() * local_mask * 0.5).sum()
            remote = torch.stack([total - local, mask.sum() - local_mask.sum()])

            def reduce(tensor, **kwargs):
                with torch.no_grad():
                    tensor.add_(remote)

            callback = functools.partial(obj._post_compute_loss, (local_mask, torch.full_like(local_mask, 0.5)))
            with (
                patch.dict(sys.modules, {"megatron.core.transformer.multi_token_prediction": stub}),
                patch.object(torch.distributed, "all_reduce", side_effect=reduce),
            ):
                loss, tokens = dispatch(
                    SimpleNamespace(),
                    (param * local_x).square(),
                    callback,
                    config,
                    None,
                    False,
                    len(masks),
                    [],
                    cp_group_size=cp,
                    is_last_stage=True,
                )
            losses.append(loss)
            counts.append(tokens)
        accumulated = sum(losses)
        if per_token:
            # finalize_model_grads sums the replicated counts across CP before clamping.
            accumulated = accumulated / (sum(counts) * cp).clamp_min(1)
        accumulated.backward()
        grads.append(param.grad)
    actual = torch.stack(grads).sum() if per_token else torch.stack(grads).mean()
    torch.testing.assert_close(actual, reference.grad)


@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("pp", [2, 4])
@pytest.mark.parametrize("per_token", [False, True])
def test_real_pipeline_callback_preserves_hidden_states_and_weights_only_final_loss(
    tp, pp, per_token, pipeline_loss_dispatch
):
    dispatch = pipeline_loss_dispatch
    config = SimpleNamespace(calculate_per_token_loss=per_token, timers=None)
    obj = trainer_class()()
    obj.args = SimpleNamespace(calculate_per_token_loss=per_token)
    obj.model = SimpleNamespace(config=SimpleNamespace(context_parallel_size=1))
    labels = torch.tensor([[-100, 1, 2, 0, -100, 1, 0, 2]])
    weights = torch.tensor([[0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5]])

    def inputs(_):
        return dict(labels=labels, loss_weights=weights)

    obj._prepare_train_inputs = inputs
    stub = SimpleNamespace(MTPLossAutoScaler=object)
    with patch.dict(sys.modules, {"megatron.core.transformer.multi_token_prediction": stub}):
        for stage in range(pp - 1):
            # The failing production tensor was [sequence/TP, batch, hidden].
            hidden = torch.randn(8 // tp, 1, 5, requires_grad=True)
            before = hidden.detach().clone()
            output, callback = obj._inner_forward_step(None, lambda **kw: hidden)
            assert output is hidden
            store = []
            output, tokens = dispatch(
                SimpleNamespace(),
                output,
                callback,
                config,
                None,
                False,
                1,
                store,
                cp_group_size=1,
                is_last_stage=False,
            )
            torch.testing.assert_close(output, before)
            assert output is hidden and tokens == 0 and store == []
            output.sum().backward()
            torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
        logits = torch.tensor([[[1.0, 2.0, 3.0]] * 8], requires_grad=True)

        def last_stage(**kw):
            return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)

        output, callback = obj._inner_forward_step(None, last_stage)
        store = []
        loss, tokens = dispatch(
            SimpleNamespace(), output, callback, config, None, False, 1, store, cp_group_size=1, is_last_stage=True
        )
        assert tokens == (labels != -100).sum()
        if per_token:
            loss = loss / tokens
        loss.backward()
        actual = logits.grad.clone()
        logits.grad = None
        (F.cross_entropy(logits.flatten(0, 1), labels.flatten()) * 0.5).backward()
        torch.testing.assert_close(actual, logits.grad)
        assert actual[0, 0].abs().sum() == actual[0, 4].abs().sum() == 0


def test_turn_and_trajectory_weights_multiply_without_losing_either():
    def collate(rows):
        return {key: torch.tensor([r[key] for r in rows]) for key in ("input_ids", "labels", "loss_weights")}

    row = dict(input_ids=[0, 1, 2], labels=[-100, 1, 2], loss_weights=[0, 0.25, 1], trajectory_loss_weight=0.5)
    out = collator_wrapper(collate)([row])
    torch.testing.assert_close(out["loss_weights"], torch.tensor([[0.125, 0.5]]))
    assert row["loss_weights"] == [0, 0.25, 1]


@pytest.mark.parametrize("mtp_layers, weight", [(0, 0.5), (1, 0.5), (1, 1.0)])
def test_mtp_rejects_fractional_weights_before_forward(mtp_layers, weight):
    cls = trainer_class()
    obj = cls.__new__(cls)
    obj.trajectory_loss_weight = True
    obj.model = SimpleNamespace(config=SimpleNamespace(mtp_num_layers=mtp_layers))
    obj._prepare_train_inputs = lambda _: dict(
        labels=torch.tensor([[-100, 1, 2]]), loss_weights=torch.tensor([[0.0, weight, weight]])
    )
    called = []

    def model(**inputs):
        called.append(inputs)
        return torch.ones(1, 3)

    if mtp_layers and weight != 1:
        with pytest.raises(ValueError, match="MTP auxiliary loss is not weighted"):
            obj._pre_compute_loss(None, model)
        assert not called
    else:
        obj._pre_compute_loss(None, model)
        assert len(called) == 1
        assert called[0]["loss_mask"].tolist() == [[0, 1, 1]]


@pytest.mark.parametrize(
    "mtp_layers,additional,resolved_mtp",
    [(0, {"mtp_num_layers": 1}, None), (1, {"mtp_num_layers": 0}, 1), (None, {}, 1), (None, {}, None)],
)
def test_trajectory_weighting_validates_resolved_mtp_before_allocation(mtp_layers, additional, resolved_mtp):
    path = ROOT / "src/llamafactory/train/mca/workflow.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "run_sft")
    config = SimpleNamespace(mtp_num_layers=resolved_mtp)
    args = SimpleNamespace(mtp_num_layers=mtp_layers, additional_configs=additional, sequence_packing=False)
    allocations = []

    class AllocationReached(Exception):
        pass

    def allocate(path, training_args, **kwargs):
        allocations.append(kwargs["config"])
        raise AllocationReached

    namespace = dict(
        Optional=Optional,
        load_tokenizer=lambda *a: {"tokenizer": object()},
        get_template_and_fix_tokenizer=lambda *a: None,
        get_dataset=lambda *a, **kw: {},
        _check_model_support=lambda *a: None,
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a: config),
        AutoModel=SimpleNamespace(from_pretrained=allocate),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    expected_error = ValueError if resolved_mtp else AllocationReached
    with pytest.raises(expected_error, match="trajectory_loss_weight requires" if resolved_mtp else None):
        namespace["run_sft"](
            SimpleNamespace(model_name_or_path="checkpoint"),
            DataArguments(trajectory_loss_weight=True),
            args,
            SimpleNamespace(),
        )
    assert allocations == ([] if resolved_mtp else [config])

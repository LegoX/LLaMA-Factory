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

"""CPU checks against the actual adapter loss code, without importing its CUDA stack."""

import ast
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "src/mcore_adapter"


def trainer_class(base):
    tree = ast.parse((ROOT / "src/llamafactory/train/mca/trainer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != "_pad_batched_inputs"]
    namespace = dict(
        McaTrainer=base, torch=torch, os=os, time=time, json=json, logger=SimpleNamespace(info=lambda *a: None)
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "trainer", "exec"), namespace)
    return namespace["CustomMcaTrainer"]


def metric_trainer(reduce=None, per_token=True, dp_group=None):
    path = ADAPTER / "trainer/trainer.py"
    methods = [
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "_reduce_loss",
            "_maybe_log_save_evaluate",
            "gather_metrics",
            "_get_step_iterator_and_seq_length",
            "_get_throughput_metrics",
        }
    ]
    namespace = dict(
        torch=torch,
        Tensor=torch.Tensor,
        Any=Any,
        Iterator=Iterator,
        Optional=Optional,
        dist=torch.distributed,
        distributed_reduce=reduce,
        mpu=SimpleNamespace(
            is_pipeline_last_stage=lambda **kw: True,
            get_data_parallel_group=lambda: dp_group,
            get_pipeline_model_parallel_world_size=lambda: 1,
        ),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    logs = []
    obj = SimpleNamespace(
        args=SimpleNamespace(calculate_per_token_loss=per_token, device="cpu"),
        model=SimpleNamespace(config=SimpleNamespace(num_moe_experts=None, mtp_num_layers=None)),
        control=SimpleNamespace(should_log=True, should_save=False, should_evaluate=False),
        state=SimpleNamespace(global_step=1),
        _globalstep_last_logged=0,
        _total_loss_scalar=0.0,
        _total_loss_tokens=0.0,
        _get_learning_rate=lambda: 0.0,
        store_flos=lambda: None,
        log=logs.append,
        metrics_keys=["loss"],
    )
    for method in methods:
        setattr(obj, method.name, namespace[method.name].__get__(obj))
    return obj, logs


@pytest.mark.parametrize("loss_sum,token_count", [(0.0, 0.0), (2.0, 1.0)])
@pytest.mark.parametrize("interval", [1, 10])
def test_per_token_logging_handles_zero_and_fractional_reduced_counts(loss_sum, token_count, interval):
    obj, logs = metric_trainer(
        lambda value, **kw: {k: v / 8 for k, v in value.items()} if isinstance(value, dict) else value / 8
    )
    obj.state.global_step = interval
    # One nonempty DP rank can leave a fractional count after the DP average.
    obj._maybe_log_save_evaluate(torch.tensor([loss_sum, token_count]), None, None, None, 0, None)
    assert logs[0]["loss"] == (loss_sum / token_count if token_count else 0.0)
    metrics = obj.gather_metrics([{"loss": (torch.tensor(0.0), torch.tensor(0.0))}])
    assert metrics["loss"] == 0.0


@pytest.mark.parametrize("per_token", [False, True])
def test_eval_metrics_reduce_before_token_normalization(per_token):
    # The second DP rank contributes 12/2 in token mode, or mean loss 6 otherwise.
    def reduce(values, **kwargs):
        remote = torch.tensor([12.0, 2.0]) if per_token else torch.tensor(6.0)
        return {key: (value + remote) / 2 for key, value in values.items()}

    obj, _ = metric_trainer(reduce, per_token)
    values = [(2.0, 1.0), (36.0, 9.0)]
    tensors = [
        {"loss": (torch.tensor(total), torch.tensor(count)) if per_token else torch.tensor(total / count)}
        for total, count in values
    ]
    assert obj.gather_metrics(tensors)["loss"] == pytest.approx(50 / 12 if per_token else 4.5)


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("interval", [1, 2, 10])
def test_train_loss_is_independent_of_logging_interval(per_token, interval):
    # Execute the production logging method and final aggregation block, including an unlogged tail.
    obj, logs = metric_trainer(lambda value, **kw: value, per_token)
    pairs = [(2.0, 1.0), (36.0, 9.0), (0.0, 0.0)]
    pending = torch.zeros(2) if per_token else torch.tensor(0.0)
    for step, (total, count) in enumerate(pairs, 1):
        pending += torch.tensor([total, count]) if per_token else total / (count or 1)
        obj.state.global_step = step
        if step % interval == 0:
            obj._maybe_log_save_evaluate(pending, None, None, None, 0, None)
    path = ADAPTER / "trainer/trainer.py"
    loop = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_inner_training_loop"
    )
    start = next(i for i, n in enumerate(loop.body) if ast.unparse(n).startswith("tr_loss = self._reduce_loss("))
    end = next(i for i, n in enumerate(loop.body) if ast.unparse(n).startswith("metrics = speed_metrics("))
    namespace = dict(self=obj, tr_loss=pending)
    exec(compile(ast.Module(body=loop.body[start:end], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["train_loss"] == pytest.approx(3.8 if per_token else 2.0)
    if interval == 2:
        assert logs[0]["loss"] == (3.8 if per_token else 3.0)


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("cp", [1, 2, 4])
def test_cp_gradients_match_unsplit_loss(cp, per_token):
    path = ADAPTER / "trainer/trainer.py"
    if not path.exists():
        pytest.skip("Set MCA_512K_ADAPTER to the patched checkout")
    tree = ast.parse(path.read_text())
    loss_method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_post_compute_loss")
    ns = dict(torch=torch, mpu=SimpleNamespace(get_context_parallel_group=lambda: None))
    exec(compile(ast.Module(body=[loss_method], type_ignores=[]), str(path), "exec"), ns)
    base = type("Base", (), {"_post_compute_loss": ns["_post_compute_loss"]})
    cls = trainer_class(base)
    trainer = cls()
    trainer.model = SimpleNamespace(config=SimpleNamespace(context_parallel_size=cp))
    trainer.args = SimpleNamespace(calculate_per_token_loss=per_token)
    x = torch.arange(1.0, 9.0)
    mask = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    reference = torch.tensor(0.3, requires_grad=True)
    ((reference * x).square() * mask).sum().div(mask.sum()).backward()
    grads = []
    for part in range(cp):
        w = torch.tensor(0.3, requires_grad=True)
        local_x = x.chunk(cp)[part]
        local_mask = mask.chunk(cp)[part]
        total_loss = ((w.detach() * x).square() * mask).sum()
        local_loss = ((w.detach() * local_x).square() * local_mask).sum()
        remote = torch.stack([total_loss - local_loss, mask.sum() - local_mask.sum()])

        def reduce(tensor, **kwargs):
            with torch.no_grad():
                tensor.add_(remote)

        with patch.object(torch.distributed, "all_reduce", side_effect=reduce):
            loss, tokens, metrics = trainer._post_compute_loss(local_mask, (w * local_x).square())
        # MCore: per-token mode sums grads then divides by DP+CP token
        # count; other mode divides microbatch loss then averages CP grads.
        denominator = tokens * cp if per_token else tokens
        (loss / denominator).backward()
        grads.append(w.grad)
        logged = metrics["loss"][0] / metrics["loss"][1] if per_token else metrics["loss"]
        torch.testing.assert_close(logged, total_loss / mask.sum())
    actual = torch.stack(grads).sum() if per_token else torch.stack(grads).mean()
    torch.testing.assert_close(actual, reference.grad)


@pytest.mark.parametrize("fails", [False, True])
def test_profile_restores_wrappers_even_after_failure(fails):
    class Base:
        def training_step(self):
            self.forward_backward_func()
            if fails:
                raise RuntimeError("synthetic failure")
            self.optimizer.step()
            return 123

    obj = trainer_class(Base)()
    obj.forward_backward_func = lambda: None
    obj.optimizer = SimpleNamespace(step=lambda: None)
    original_forward, original_step = obj.forward_backward_func, obj.optimizer.step
    obj.state = SimpleNamespace(global_step=0)
    obj.is_world_process_zero = lambda: True
    with patch.dict(os.environ, LF_MCA_PROFILE="1"), patch.object(torch.cuda, "is_available", return_value=False):
        if fails:
            with pytest.raises(RuntimeError, match="synthetic failure"):
                obj.training_step()
        else:
            assert obj.training_step() == 123
            assert obj._profile_metrics["forward_backward_seconds"] >= 0
            assert obj._profile_metrics["optimizer_seconds"] >= 0
    assert obj.forward_backward_func is original_forward
    assert obj.optimizer.step is original_step
    assert not obj._profile_active


@pytest.mark.parametrize("micro_batch", [1, 2])
@pytest.mark.parametrize("variable_lengths", [False, True])
def test_throughput_counts_accumulated_inputs_after_padding(micro_batch, variable_lengths):
    obj, _ = metric_trainer()
    obj.model.main_input_name = "input_ids"
    obj.args.gradient_accumulation_steps = 2
    obj.args.per_device_train_batch_size = micro_batch
    obj.args.allow_variable_seq_lengths = lambda: variable_lengths
    obj.current_flos = 0
    obj.floating_point_ops = lambda inputs: 0
    obj._pad_batched_inputs = lambda inputs, length: {
        "input_ids": torch.nn.functional.pad(inputs["input_ids"], (0, length - inputs["input_ids"].size(1)))
    }
    batches = [{"input_ids": torch.ones(micro_batch, length, dtype=torch.long)} for length in (2, 5)]
    iterator, max_length, total_length = obj._get_step_iterator_and_seq_length(iter(batches))
    expected_tokens = micro_batch * (7 if variable_lengths else 10)
    assert sum(batch["input_ids"].numel() for batch in iterator) == expected_tokens
    assert obj._step_input_tokens == expected_tokens
    assert (max_length, total_length) == (5, 7)  # Preserve the legacy throughput numerator.


@pytest.mark.parametrize("world_size", [16, 32])
@pytest.mark.parametrize("elapsed,remote_elapsed", [(1.0, 2.0), (2.0, 1.0)])
def test_throughput_sums_unequal_dp_inputs_over_slowest_rank_time(world_size, elapsed, remote_elapsed):
    dp_group = object()
    obj, _ = metric_trainer(dp_group=dp_group)
    obj._step_input_tokens = 100

    def reduce(value, op, group=None):
        if op == torch.distributed.ReduceOp.SUM:
            assert group is dp_group  # Do not count duplicate TP/PP/CP inputs.
            value.add_(1000)
        else:
            assert op == torch.distributed.ReduceOp.MAX and group is None
            value.fill_(max(value.item(), remote_elapsed))

    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=world_size),
        patch.object(torch.distributed, "all_reduce", side_effect=reduce) as reductions,
    ):
        metrics = obj._get_throughput_metrics(elapsed)
    assert reductions.call_count == 2
    assert metrics["input_tokens_per_second"] == 550.0
    assert metrics["input_tokens_per_second_per_gpu"] == 550.0 / world_size


def test_throughput_without_distributed_initialization():
    obj, _ = metric_trainer()
    obj._step_input_tokens = 100
    with patch.object(torch.distributed, "is_initialized", return_value=False):
        metrics = obj._get_throughput_metrics(2.0)
    assert metrics == {"input_tokens_per_second": 50.0, "input_tokens_per_second_per_gpu": 50.0}


def test_profile_preserves_adapter_throughput_metrics():
    class Base:
        def _maybe_log_save_evaluate(self, **kwargs):
            return kwargs["other_logs"]

    obj = trainer_class(Base)()
    obj._profile_metrics = {"optimizer_seconds": 2.0}
    logs = {
        "token_per_sec_per_gpu": 1600.0,
        "input_tokens_per_second": 550.0,
        "input_tokens_per_second_per_gpu": 34.375,
    }
    assert obj._maybe_log_save_evaluate(other_logs=logs) == {**logs, "optimizer_seconds": 2.0}
    assert "optimizer_seconds" not in logs

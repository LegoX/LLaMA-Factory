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
from types import SimpleNamespace

from transformers import TrainerControl, TrainerState

from llamafactory.train.callbacks import LogCallback


def test_log_callback_persists_metrics_without_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMABOARD_ENABLED", raising=False)
    monkeypatch.delenv("RECORD_VRAM", raising=False)
    args = SimpleNamespace(should_save=True, output_dir=str(tmp_path))
    state = TrainerState(max_steps=48, global_step=1)
    control = TrainerControl()
    metrics = {
        "loss": 0.746,
        "grad_norm": 2.337578773498535,
        "learning_rate": 1e-5,
        "epoch": 0.0625,
        "step": 1,
        "forward_backward_seconds": 1515.774254471995,
        "optimizer_seconds": 51.08927187137306,
        "training_step_seconds": 1518.8780025551096,
        "input_tokens_per_second": 8886.212082236512,
        "input_tokens_per_second_per_gpu": 555.388255139782,
        "peak_cuda_allocated_gib": 77.0997486114502,
        "skipped_iter": 0,
        "num_zeros_in_grad": 0,
    }
    callback = LogCallback()
    callback.on_train_begin(args, state, control)
    try:
        state.log_history.append(metrics.copy())
        callback.on_log(args, state, control)
        state.log_history.append({"eval_loss": 0.6, "step": 1})
        callback.on_log(args, state, control)
    finally:
        callback.on_train_end(args, state, control)

    rows = [json.loads(line) for line in (tmp_path / "trainer_log.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert metrics.items() <= rows[0].items()
    assert state.log_history[0] == metrics
    assert rows[0]["lr"] == metrics["learning_rate"]
    assert rows[0]["current_steps"] == 1
    assert rows[0]["total_steps"] == 48
    assert rows[0]["percentage"] == 2.08
    assert "elapsed_time" in rows[0]
    assert rows[1]["eval_loss"] == 0.6
    assert "grad_norm" not in rows[1]  # Evaluation must not inherit stale training metrics.
    assert "loss" not in rows[1]


def test_log_callback_does_not_write_from_non_saving_rank(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMABOARD_ENABLED", raising=False)
    output_dir = tmp_path / "non_saving_rank"
    args = SimpleNamespace(should_save=False, output_dir=str(output_dir))
    state = TrainerState(max_steps=48, global_step=1, log_history=[{"grad_norm": 2.3}])
    control = TrainerControl()
    callback = LogCallback()
    callback.on_train_begin(args, state, control)
    try:
        callback.on_log(args, state, control)
    finally:
        callback.on_train_end(args, state, control)
    assert not output_dir.exists()

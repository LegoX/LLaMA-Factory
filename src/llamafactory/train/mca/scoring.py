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

"""Pre-update, teacher-forced scoring using MCA's TP/PP forward-only schedule.

Each microbatch contains one dataset row (CP1/DP1). Multiple microbatches
fill the pipeline without padding different rows to a common sequence length.
The training sampler is not used, since it may repeat rows to fill steps.
"""

import json
import math
import os
import time
from pathlib import Path

import torch

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger


logger = get_logger(__name__)


def validate_scoring_config(config, args, data_args):
    """Validate the resolved model configuration before allocating weights."""
    tp = config.tensor_model_parallel_size
    pp = config.pipeline_model_parallel_size
    if config.context_parallel_size != 1 or int(os.environ.get("WORLD_SIZE", "1")) != tp * pp:
        raise ValueError("pretrain_score currently requires CP1/DP1 (WORLD_SIZE = TP * PP).")
    if data_args.packing or args.sequence_packing or config.virtual_pipeline_model_parallel_size:
        raise ValueError("pretrain_score does not support packing or virtual pipeline stages.")
    if args.resume_from_checkpoint:
        raise ValueError("pretrain_score scores initial weights only; checkpoint resume is not supported.")
    if data_args.pretrain_score_microbatches > 1 and pp > 1 and not config.variable_seq_lengths:
        raise ValueError("Pipelined scoring requires variable_seq_lengths=true or pretrain_score_microbatches=1.")


def _statistics(values):
    count = values.numel()
    if count == 0:
        return {"num_loss_tokens": 0, "loss": None, "ppl": None, "status": "no_valid_tokens"}
    loss = values.double().mean().item()
    if not math.isfinite(loss):
        return {"num_loss_tokens": count, "loss": None, "ppl": None, "status": "nonfinite_loss"}
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = None
    return {"num_loss_tokens": count, "loss": loss, "ppl": ppl, "status": "ok" if ppl is not None else "ppl_overflow"}


def aggregate_sample(losses, labels, metadata, sample_index, padding_side="right"):
    """Aggregate already-shifted MCA token losses; spans refer to unshifted tokens."""
    if labels.ndim != 2 or labels.shape[0] != 1 or losses.shape != labels.shape:
        raise ValueError("Expected batch=1 token losses with the same [batch, sequence] shape as MCA labels.")
    # Transfer once, then aggregate in float64 on CPU. Never serialize NaN/Infinity.
    losses = losses.detach().float().cpu()[0]
    labels = labels.detach().cpu()[0]
    offset = labels.numel() - (metadata["retained_length"] - 1) if padding_side == "left" else 0
    if offset < 0:
        raise ValueError("Scoring labels are shorter than the retained sequence.")
    valid = labels != IGNORE_INDEX
    covered = torch.zeros_like(valid)
    turns = []
    for turn in metadata["turns"]:
        start, end = turn["start"], turn["end"]
        if start is None:
            values = losses[:0]
        else:
            # The collator has shifted labels by one, including the final target token.
            start, end = max(start - 1, 0) + offset, max(end - 1, 0) + offset
            if end > labels.numel() or start > end or covered[start:end].any():
                raise ValueError("Invalid or overlapping assistant turn boundaries.")
            covered[start:end] = True
            values = losses[start:end][valid[start:end]]
        turns.append({**turn, **_statistics(values)})
    if (valid & ~covered).any():
        raise ValueError("Some supervised tokens have no assistant turn; refusing misleading turn-level scores.")
    return {
        "sample_index": sample_index,
        "sample_id": json.dumps([metadata["dataset"], metadata["split"], metadata["source_row_index"]]),
        **{key: value for key, value in metadata.items() if key != "turns"},
        **_statistics(losses[valid]),
        "turns": turns,
    }


def score_training_dataset(trainer, model_name_or_path, num_microbatches=16):
    from megatron.core import parallel_state as mpu

    if mpu.get_context_parallel_world_size() != 1 or mpu.get_data_parallel_world_size() != 1:
        raise ValueError("pretrain_score currently requires CP1/DP1.")
    models = trainer.model.get_models()
    if len(models) != 1 or trainer.model.config.mtp_num_layers:
        raise ValueError("pretrain_score requires one model chunk per rank and mtp_num_layers=0.")
    if isinstance(num_microbatches, bool) or not isinstance(num_microbatches, int) or num_microbatches < 1:
        raise ValueError("pretrain_score_microbatches must be a positive integer.")
    if (
        num_microbatches > 1
        and mpu.get_pipeline_model_parallel_world_size() > 1
        and not trainer.model.config.variable_seq_lengths
    ):
        raise ValueError(
            "Pipelined scoring requires variable_seq_lengths=true; otherwise use pretrain_score_microbatches=1."
        )
    dataset = trainer.train_dataset
    if not dataset or "score_metadata" not in dataset.column_names:
        raise ValueError("Missing scoring metadata; rebuild the dataset with pretrain_score=true.")
    tokenizer = trainer.processing_class
    padding_side = getattr(tokenizer, "padding_side", "right")
    writer = mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0
    writer_rank = torch.tensor(torch.distributed.get_rank() if writer else -1, device=trainer.args.device)
    torch.distributed.all_reduce(writer_rank, op=torch.distributed.ReduceOp.MAX)
    writer_rank = writer_rank.item()
    output_dir = Path(trainer.args.output_dir)
    output = output_dir / "pretrain_scores.jsonl"
    incomplete = output_dir / "pretrain_scores.jsonl.incomplete"
    handle = None

    def writer_action(action):
        # Propagate writer I/O/aggregation errors to every stage before its next forward.
        error = [None]
        if writer:
            try:
                action()
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        torch.distributed.broadcast_object_list(error, src=writer_rank)
        if error[0]:
            raise RuntimeError(f"Pretrain scoring failed: {error[0]}")

    def open_output():
        nonlocal handle
        output_dir.mkdir(parents=True, exist_ok=True)
        if output.exists() or incomplete.exists():
            raise FileExistsError(f"Scoring output already exists in {output_dir}; choose a fresh output_dir.")
        handle = incomplete.open("x", encoding="utf-8")

    modes = [model.training for model in models]
    logger.info_rank0(f"Scoring {len(dataset)} training rows before optimizer updates; output: {output}")
    try:
        writer_action(open_output)
        for model in models:
            model.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            for block_start in range(0, len(dataset), num_microbatches):
                started = time.perf_counter()
                block_end = min(block_start + num_microbatches, len(dataset))
                # Collate on CPU. Transfer just the current microbatch to GPU when
                # its stage executes; never retain a block of long input tensors on GPU.
                batches = []
                for index in range(block_start, block_end):
                    row = dataset[index]
                    metadata = json.loads(row["score_metadata"])
                    batches.append((index, metadata, trainer.data_collator([row])))
                pending = []

                def forward_step(data_iterator, model):
                    index, metadata, batch = next(data_iterator)
                    batch = trainer._prepare_inputs(batch)
                    inputs = trainer._prepare_train_inputs(iter([batch]))
                    labels = inputs["labels"]
                    inputs.pop("loss_weights", None)
                    inputs["loss_mask"] = (labels != IGNORE_INDEX).float()
                    output_tensor = model(**inputs)

                    def loss_callback(losses):
                        if writer:
                            # Defer CPU transfers/aggregation until the pipeline drains.
                            # Keep only O(sequence length) losses/labels, never logits or
                            # activations. Each closure owns its row's metadata/labels.
                            pending.append((index, metadata, losses.detach(), labels.detach()))
                        return losses.new_zeros(()), (labels != IGNORE_INDEX).sum().int().clamp_min(1), {}

                    return output_tensor, loss_callback

                trainer.forward_backward_func(
                    forward_step_func=forward_step,
                    data_iterator=[iter(batches)],
                    model=models,
                    num_microbatches=len(batches),
                    seq_length=max(batch["input_ids"].shape[1] for _, _, batch in batches),
                    micro_batch_size=1,
                    forward_only=True,
                )

                def write_block():
                    if [item[0] for item in pending] != list(range(block_start, block_end)):
                        raise RuntimeError("Final-stage results are missing, duplicated or out of order.")
                    for index, metadata, losses, labels in pending:
                        result = aggregate_sample(losses, labels, metadata, index, padding_side)
                        handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()

                writer_action(write_block)
                pending.clear()
                elapsed = time.perf_counter() - started
                input_tokens = sum(metadata["retained_length"] - 1 for _, metadata, _ in batches)
                logger.info_rank0(
                    f"Pretrain scoring: {block_end}/{len(dataset)} rows; "
                    f"microbatches={len(batches)} elapsed={elapsed:.2f}s "
                    f"input_tokens_per_second={input_tokens / elapsed:.1f}"
                )

        def finish():
            handle.close()
            manifest = {
                "num_samples": len(dataset),
                "scoring_microbatches": num_microbatches,
                "model_name_or_path": model_name_or_path,
                "scoring": "initial_weights_teacher_forcing",
                "assistant_tokens_only": True,
                "sample_loss": "token_weighted_mean_cross_entropy",
                "turn_ppl": "exp(turn_mean_cross_entropy)",
                "source_row_index_base": 0,
                "turn_id_base": 0,
                "writer_global_rank": writer_rank,
                "tensor_parallel_size": mpu.get_tensor_model_parallel_world_size(),
                "pipeline_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
            }
            (output_dir / "pretrain_scores.meta.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            incomplete.replace(output)

        writer_action(finish)
        logger.info_rank0(f"Pretrain scoring complete: {output}")
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        for model, mode in zip(models, modes):
            model.train(mode)

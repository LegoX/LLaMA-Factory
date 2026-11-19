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

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from transformers import DataCollatorWithPadding, LlamaConfig, LlamaForCausalLM

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import FinetuningArguments, TrainingArguments, get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer, _compute_weighted_loss


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "max_steps": 1,
    "report_to": "none",
}


@dataclass
class DataCollatorWithVerbose(DataCollatorWithPadding):
    verbose_list: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [
            {k: v for k, v in feature.items() if k in ["input_ids", "attention_mask", "labels"]}
            for feature in features
        ]
        self.verbose_list.extend(features)
        batch = super().__call__(features)
        return {k: v[:, :1] for k, v in batch.items()}  # truncate input length


def test_compute_weighted_loss():
    logits = torch.tensor(
        [
            [
                [0.0, 2.0, -1.0],
                [0.0, -1.0, 2.0],
                [0.0, 2.0, -1.0],
                [2.0, 0.0, -1.0],
            ]
        ]
    )
    labels = torch.tensor([[IGNORE_INDEX, 1, 2, 1]])
    loss_weights = torch.tensor([[0.0, 1.0, 0.5, 0.0]])
    loss = _compute_weighted_loss({"logits": logits}, labels, loss_weights)

    per_token_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    expected_weights = torch.tensor([1.0, 0.5, 0.0])
    expected = (per_token_loss * expected_weights).sum() / (labels[:, 1:] != IGNORE_INDEX).sum()
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("disable_shuffling", [False, True])
def test_shuffle(disable_shuffling: bool):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        {
            "output_dir": os.path.join("output", f"shuffle{str(disable_shuffling).lower()}"),
            "disable_shuffling": disable_shuffling,
            **TRAIN_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorWithVerbose(tokenizer=tokenizer)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )
    trainer.train()
    if disable_shuffling:
        assert data_collator.verbose_list[0]["input_ids"] == dataset_module["train_dataset"][0]["input_ids"]
    else:
        assert data_collator.verbose_list[0]["input_ids"] != dataset_module["train_dataset"][0]["input_ids"]


def test_uniform_weight_scales_hf_loss_gradient():
    labels = torch.tensor([[-100, 1, 2]])
    logits = torch.randn(1, 3, 4, requires_grad=True)
    plain = _compute_weighted_loss({"logits": logits}, labels, torch.ones_like(labels).float())
    plain.backward()
    expected = logits.grad.clone() * 0.25
    logits.grad = None
    weighted = _compute_weighted_loss({"logits": logits}, labels, torch.full_like(labels, 0.25, dtype=torch.float32))
    weighted.backward()
    torch.testing.assert_close(logits.grad, expected)


@pytest.mark.parametrize("processor", [None, object()])
@pytest.mark.parametrize("second_labels", [[-100, 3, 4, 5], [-100, -100, -100, 5], [-100] * 4])
@pytest.mark.parametrize("configured_accumulation", [2, 4])
def test_weighted_trainer_gradient_accumulation(tmp_path, processor, second_labels, configured_accumulation):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0,
            use_cache=False,
        )
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
            gradient_accumulation_steps=configured_accumulation,
        ),
        finetuning_args=FinetuningArguments(finetuning_type="full"),
        processor=processor,
        tokenizer=None,
    )
    trainer.create_optimizer()
    # Also exercise a final, partial accumulation window (2 actual vs 4 configured).
    trainer.current_gradient_accumulation_steps = 2
    batches = [
        dict(
            input_ids=torch.tensor([[2, 3, 4, 5]]),
            labels=torch.tensor([labels]),
            loss_weights=torch.tensor([[0.0, 1.0, 0.5, 0.25]]),
        )
        for labels in [[-100, 3, 4, 5], second_labels]
    ]
    batches, count = trainer.get_batch_samples(iter(batches), len(batches), torch.device("cpu"))
    for batch in batches:
        trainer.training_step(model, deepcopy(batch), num_items_in_batch=count)
    accumulated = {name: param.grad.clone() for name, param in model.named_parameters()}
    model.zero_grad()
    # Independent reference: weighted token sum / valid tokens, averaged over microbatches.
    for batch in batches:
        logits = model(input_ids=batch["input_ids"]).logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
        loss = (ce * batch["loss_weights"][:, 1:].flatten()).sum() / (labels != -100).sum().clamp_min(1)
        (loss / len(batches)).backward()
    for name, param in model.named_parameters():
        torch.testing.assert_close(accumulated[name], param.grad)

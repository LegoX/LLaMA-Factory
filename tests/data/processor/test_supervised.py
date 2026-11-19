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
import random

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.hparams import DataArguments, get_infer_args
from llamafactory.model import load_tokenizer
from llamafactory.train.test_utils import load_dataset_module


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TINY_DATA = os.getenv("TINY_DATA", "llamafactory/tiny-supervised-dataset")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "template": "llama3",
    "cutoff_len": 8192,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("mask_history", [False, True])
def test_supervised_turn_mask(mask_history: bool):
    class DummyMMPlugin:
        def process_messages(self, messages, images, videos, audios, processor):
            return messages

        def process_token_ids(self, input_ids, labels, images, videos, audios, tokenizer, processor):
            return input_ids, labels

    class DummyTemplate:
        efficient_eos = False
        preserve_thinking = True
        mm_plugin = DummyMMPlugin()

        def encode_multiturn(self, tokenizer, messages, system, tools, discarding_history_cot):
            return [([10], [11, 12]), ([20], [21, 22])]

    class DummyTokenizer:
        eos_token_id = 2

    data_args = DataArguments(cutoff_len=100, mask_history=mask_history, turn_mask=True)
    dataset_processor = SupervisedDatasetProcessor(
        template=DummyTemplate(),
        tokenizer=DummyTokenizer(),
        processor=None,
        data_args=data_args,
    )
    input_ids, labels, _, full_length = dataset_processor._encode_data_example(
        prompt=[
            {"role": "user", "content": "first question", "turn_mask": False},
            {"role": "assistant", "content": "first answer", "turn_mask": True},
            {"role": "user", "content": "second question", "turn_mask": False},
        ],
        response=[{"role": "assistant", "content": "second answer", "turn_mask": False}],
        system=None,
        tools=None,
        images=[],
        videos=[],
        audios=[],
    )

    assert input_ids == [10, 11, 12, 20, 21, 22]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 21, 22]
    assert full_length == 6


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("num_samples", [16])
def test_supervised_single_turn(num_samples: int):
    train_dataset = load_dataset_module(dataset_dir="ONLINE", dataset=TINY_DATA, **TRAIN_ARGS)["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(TINY_DATA, split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        prompt = original_data["instruction"][index]
        if original_data["input"][index]:
            prompt += "\n" + original_data["input"][index]

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": original_data["output"][index]},
        ]
        ref_input_ids = ref_tokenizer.apply_chat_template(messages)
        ref_prompt_ids = ref_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)

        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]
            ref_prompt_ids = ref_prompt_ids["input_ids"]

        prompt_len = len(ref_prompt_ids)
        ref_label_ids = [IGNORE_INDEX] * prompt_len + ref_input_ids[prompt_len:]
        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_label_ids


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("num_samples", [8])
def test_supervised_multi_turn(num_samples: int):
    train_dataset = load_dataset_module(dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", **TRAIN_ARGS)[
        "train_dataset"
    ]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        ref_input_ids = ref_tokenizer.apply_chat_template(original_data["messages"][index])
        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]

        # cannot test the label ids in multi-turn case
        assert train_dataset["input_ids"][index] == ref_input_ids


@pytest.mark.runs_on(["cpu", "mps"])
def test_supervised_loss_weights():
    model_args, data_args, *_ = get_infer_args(
        {"model_name_or_path": TINY_LLAMA3, "template": "llama3", "cutoff_len": 8192}
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
        data_args=data_args,
    )
    model_inputs = processor.preprocess_dataset(
        {
            "_prompt": [
                [
                    {"role": "user", "content": "First request."},
                    {"role": "assistant", "content": "Weighted answer.", "loss_weight": 0.5},
                    {"role": "user", "content": "Second request."},
                ]
            ],
            "_response": [[{"role": "assistant", "content": "Masked answer.", "loss_weight": 0.0}]],
            "_system": [""],
            "_tools": [""],
            "_images": [None],
            "_videos": [None],
            "_audios": [None],
        }
    )
    example = {key: value[0] for key, value in model_inputs.items()}
    assert len(example["loss_weights"]) == len(example["input_ids"]) == len(example["labels"])
    assert all(
        weight == 0.0 for label, weight in zip(example["labels"], example["loss_weights"]) if label == IGNORE_INDEX
    )

    valid_weights = [
        round(float(weight), 2)
        for label, weight in zip(example["labels"], example["loss_weights"])
        if label != IGNORE_INDEX
    ]
    assert valid_weights
    assert set(valid_weights) == {0.5}

    decoded_labels = tokenizer.decode([label for label in example["labels"] if label != IGNORE_INDEX])
    assert "Weighted answer" in decoded_labels
    assert "Masked answer" not in decoded_labels


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("num_samples", [4])
def test_supervised_train_on_prompt(num_samples: int):
    train_dataset = load_dataset_module(
        dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", train_on_prompt=True, **TRAIN_ARGS
    )["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        ref_input_ids = ref_tokenizer.apply_chat_template(original_data["messages"][index])
        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]

        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_input_ids


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("num_samples", [4])
def test_supervised_mask_history(num_samples: int):
    train_dataset = load_dataset_module(
        dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", mask_history=True, **TRAIN_ARGS
    )["train_dataset"]
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        messages = original_data["messages"][index]
        ref_input_ids = ref_tokenizer.apply_chat_template(messages)
        ref_prompt_ids = ref_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)

        if is_transformers_version_greater_than("5.0.0"):
            ref_input_ids = ref_input_ids["input_ids"]
            ref_prompt_ids = ref_prompt_ids["input_ids"]

        prompt_len = len(ref_prompt_ids)
        ref_label_ids = [IGNORE_INDEX] * prompt_len + ref_input_ids[prompt_len:]
        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_label_ids

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
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoProcessor, AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.template import TEMPLATES
from llamafactory.hparams import DataArguments


@pytest.fixture
def text_encoder():
    # Keep formatting regressions runnable without downloading a model tokenizer.
    return SimpleNamespace(encode=lambda text, **kwargs: list(map(ord, text)))


@pytest.mark.parametrize("multiturn", [False, True])
@pytest.mark.parametrize("thought", ["<think>secret</think>", "  <think>\nsecret\n</think>\n\n"])
def test_disabled_thinking_removes_complete_prefix(text_encoder, multiturn, thought):
    template = deepcopy(TEMPLATES["qwen3_8"])
    template.enable_thinking = False
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": thought + "answer"}]
    original = deepcopy(messages)
    source, target = (
        template.encode_multiturn(text_encoder, messages)[0]
        if multiturn
        else template.encode_oneturn(text_encoder, messages)
    )
    assert "".join(map(chr, source)).endswith("<think>\n\n</think>\n\n")
    assert "".join(map(chr, target)) == "answer<|im_end|>\n"
    assert messages == original


@pytest.mark.parametrize("multiturn", [False, True])
@pytest.mark.parametrize("role", ["assistant", "function"])
@pytest.mark.parametrize("literal", ["Print <think> literally.", "Print </think> literally.", "<think>unfinished"])
def test_literal_tags_do_not_suppress_empty_thinking(text_encoder, multiturn, role, literal):
    template = deepcopy(TEMPLATES["qwen3_8"])
    content = literal if role == "assistant" else json.dumps({"name": "echo", "arguments": {"text": literal}})
    messages = [{"role": "user", "content": "Hi"}, {"role": role, "content": content}]
    _, target = (
        template.encode_multiturn(text_encoder, messages)[0]
        if multiturn
        else template.encode_oneturn(text_encoder, messages)
    )
    text = "".join(map(chr, target))
    assert text.startswith("<think>\n\n</think>\n\n")
    assert literal in text


@pytest.mark.parametrize("multiturn", [False, True])
def test_discard_history_removes_inline_thought_only_from_history(text_encoder, multiturn):
    template = deepcopy(TEMPLATES["qwen3_8"])
    template.preserve_thinking = False
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "<think>old secret</think>first answer"},
        {"role": "user", "content": "Second"},
        {"role": "assistant", "content": "<think>new thought</think>second answer"},
    ]
    original = deepcopy(messages)
    if multiturn:
        pairs = template.encode_multiturn(text_encoder, messages, discarding_history_cot=True)
    else:
        pairs = [template.encode_oneturn(text_encoder, messages)]
    text = "".join(chr(token) for source, target in pairs for token in source + target)
    assert "old secret" not in text
    assert "first answer" in text and "<think>\nnew thought\n</think>\n\nsecond answer" in text
    assert messages == original


@pytest.mark.parametrize("multiturn", [False, True])
@pytest.mark.parametrize("wrapped_response", [False, True])
def test_discard_history_preserves_current_tool_chain(text_encoder, multiturn, wrapped_response):
    template = deepcopy(TEMPLATES["qwen3_8"])
    template.preserve_thinking = False
    call = json.dumps({"name": "terminal", "arguments": {"command": "pwd"}})
    observation = (
        {"role": "user", "content": "  <tool_response>\n/work\n</tool_response>  "}
        if wrapped_response
        else {"role": "observation", "content": "/work"}
    )
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "function", "content": "<think>old tool reasoning</think>" + call},
        observation,
        {"role": "assistant", "content": "<think>old answer reasoning</think>first answer"},
        {"role": "user", "content": "Second question"},
        {"role": "function", "content": "<think>current tool reasoning</think>" + call},
        observation,
        {"role": "function", "content": call},
        observation,
        {"role": "assistant", "content": "<think>current answer reasoning</think>second answer"},
    ]
    original = deepcopy(messages)
    pairs = (
        template.encode_multiturn(text_encoder, messages, discarding_history_cot=True)
        if multiturn
        else [template.encode_oneturn(text_encoder, messages)]
    )
    text = "".join(chr(token) for source, target in pairs for token in source + target)
    assert "old tool reasoning" not in text and "old answer reasoning" not in text
    assert "<think>\ncurrent tool reasoning\n</think>" in text
    assert "<think>\ncurrent answer reasoning\n</think>" in text
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>" in text
    assert text.count("<think>") == 3
    assert messages == original


def official_messages(messages):
    """Separate LF's embedded thinking into the official OpenAI message fields."""
    result = []
    for message in messages:
        message = dict(message)
        if message["role"] == "assistant" and message["content"].startswith("<think>"):
            thought, separator, answer = message["content"][len("<think>") :].partition("</think>")
            assert separator
            message["reasoning_content"] = thought
            message["content"] = answer
        result.append(message)
    return result


@pytest.fixture
def tokenizer():
    model = os.getenv("QWEN38_TOKENIZER", "")
    if not model or not Path(model).is_dir():
        pytest.skip("Set QWEN38_TOKENIZER to a local Qwen3.8 tokenizer")
    return AutoTokenizer.from_pretrained(model, local_files_only=True)


@pytest.mark.parametrize("effort", ["xhigh", "medium", "low"])
@pytest.mark.parametrize("enable_thinking", [True, None])
def test_official_template_parity(tokenizer, effort, enable_thinking):
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Compute 1+1."},
        {"role": "assistant", "content": "<think>\nAdd the integers.\n</think>\n\n2"},
        {"role": "user", "content": "And 2+2?"},
        {"role": "assistant", "content": "<think>\nAdd again.\n</think>\n\n4"},
    ]
    expected = tokenizer.apply_chat_template(
        official_messages(messages),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        reasoning_effort=effort,
        preserve_thinking=True,
    )
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    template = get_template_and_fix_tokenizer(
        tokenizer, DataArguments(template="qwen3_8", reasoning_effort=effort, enable_thinking=enable_thinking)
    )
    pairs = template.encode_multiturn(tokenizer, messages[1:], system=messages[0]["content"])
    actual = [token for source, target in pairs for token in source + target]
    assert actual == expected
    assert template.preserve_thinking is True


@pytest.mark.parametrize("enable_thinking", [True, False, None])
@pytest.mark.parametrize("preserve_thinking", [False, True])
@pytest.mark.parametrize("effort", ["xhigh", "medium", "low"])
def test_official_generation_prompt(tokenizer, enable_thinking, preserve_thinking, effort):
    messages = [
        {"role": "user", "content": "Compute 1+1."},
        {"role": "assistant", "content": "<think>Add the integers.</think>2"},
        {"role": "user", "content": "And 2+2?"},
    ]
    original = deepcopy(messages)
    official = official_messages(messages)
    if enable_thinking is False:
        # LF's disabled-thinking mode also removes reasoning from supplied history.
        official[1]["reasoning_content"] = ""
    expected = tokenizer.apply_chat_template(
        official,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
        reasoning_effort=effort,
    )
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    template = get_template_and_fix_tokenizer(
        tokenizer,
        DataArguments(
            template="qwen3_8",
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
            reasoning_effort=effort,
        ),
    )
    assert template.encode_generation(tokenizer, messages) == expected
    assert messages == original


@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_generation_does_not_change_empty_training_response(text_encoder, enable_thinking):
    template = deepcopy(TEMPLATES["qwen3_8"])
    template.enable_thinking = enable_thinking
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}]
    source, target = template.encode_oneturn(text_encoder, messages)
    source_text, target_text = "".join(map(chr, source)), "".join(map(chr, target))
    empty_thought = "<think>\n\n</think>\n\n"
    assert source_text.endswith("<|im_start|>assistant\n" + ("" if enable_thinking else empty_thought))
    assert target_text == (empty_thought if enable_thinking else "") + "<|im_end|>\n"
    prompt = "".join(map(chr, template.encode_generation(text_encoder, messages[:-1])))
    assert prompt.endswith("<|im_start|>assistant\n" + (empty_thought if enable_thinking is False else "<think>\n"))


@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("thought", ["", "  Check carefully.\n"])
@pytest.mark.parametrize("answer", ["  Hi\n", " \n "])
def test_message_whitespace_matches_official_template(tokenizer, enable_thinking, thought, answer):
    official = [
        {"role": "user", "content": "  Hello\n"},
        {"role": "assistant", "content": answer, "reasoning_content": thought if enable_thinking else ""},
    ]
    content = f"  <think>\n{thought}\n</think>\n\n{answer}" if thought else answer
    messages = [official[0], {"role": "assistant", "content": content}]
    expected = tokenizer.apply_chat_template(
        official, tokenize=True, add_generation_prompt=False, reasoning_effort="medium", preserve_thinking=True
    )
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    template = get_template_and_fix_tokenizer(
        tokenizer, DataArguments(template="qwen3_8", reasoning_effort="medium", enable_thinking=enable_thinking)
    )
    pairs = template.encode_multiturn(tokenizer, messages)
    assert [token for source, target in pairs for token in source + target] == expected
    source, target = template.encode_oneturn(tokenizer, messages)
    assert source + target == expected


@pytest.mark.parametrize("effort", ["xhigh", "medium", "low"])
@pytest.mark.parametrize("system", [None, "Be precise."])
def test_empty_tools_match_official_template(tokenizer, effort, system):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "<think>\nRespond.\n</think>\n\nHi"},
    ]
    official = ([{"role": "system", "content": system}] if system else []) + official_messages(messages)
    expected = tokenizer.apply_chat_template(
        official, tools=[], tokenize=True, add_generation_prompt=False, reasoning_effort=effort, preserve_thinking=True
    )
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen3_8", reasoning_effort=effort))
    pairs = template.encode_multiturn(tokenizer, messages, system=system, tools="[]")
    assert [token for source, target in pairs for token in source + target] == expected


def test_official_tool_system_parity(tokenizer):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a command.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    messages = [
        {"role": "system", "content": "Use local tools."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "<think>\nRespond.\n</think>\n\nHi"},
    ]
    expected = tokenizer.apply_chat_template(
        official_messages(messages),
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        reasoning_effort="medium",
        preserve_thinking=True,
    )
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen3_8", reasoning_effort="medium"))
    pairs = template.encode_multiturn(tokenizer, messages[1:], system=messages[0]["content"], tools=json.dumps(tools))
    assert [token for source, target in pairs for token in source + target] == expected


def test_invalid_effort():
    with pytest.raises(ValueError, match="reasoning_effort"):
        DataArguments(reasoning_effort="invalid")


def test_incompatible_tool_format(tokenizer):
    with pytest.raises(ValueError, match="built-in tool format"):
        get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen3_8", tool_format="qwen"))


@pytest.mark.parametrize("masks", [(True, False), (False, True), (True, True)])
def test_multiturn_supervision_matches_template(tokenizer, masks):
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.extras.constants import IGNORE_INDEX

    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Compute 1+1."},
        {"role": "assistant", "content": "<think>\nAdd.\n</think>\n\n2", "turn_mask": masks[0]},
        {"role": "user", "content": "And 2+2?"},
        {"role": "assistant", "content": "<think>\nAdd.\n</think>\n\n4", "turn_mask": masks[1]},
    ]
    args = DataArguments(
        template="qwen3_8",
        reasoning_effort="medium",
        preserve_thinking=True,
        turn_mask=True,
        cutoff_len=2_000_000,
    )
    template = get_template_and_fix_tokenizer(tokenizer, args)
    system = messages[0]["content"]
    body = messages[1:]
    expected = tokenizer.apply_chat_template(
        official_messages(messages),
        tokenize=True,
        add_generation_prompt=False,
        reasoning_effort="medium",
        preserve_thinking=True,
    )
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    pairs = template.encode_multiturn(tokenizer, body, system=system)
    assert [token for source, target in pairs for token in source + target] == expected
    model_processor = AutoProcessor.from_pretrained(tokenizer.name_or_path, local_files_only=True)
    processor = SupervisedDatasetProcessor(
        template=template, tokenizer=tokenizer, processor=model_processor, data_args=args
    )
    ids, labels, _, full_length = processor._encode_data_example(
        prompt=body[:-1],
        response=body[-1:],
        system=system,
        tools=None,
        images=[],
        videos=[],
        audios=[],
    )
    assert ids == expected
    assert full_length == len(ids)
    position = 0
    for (source, target), message in zip(pairs, body[1::2]):
        assert labels[position : position + len(source)] == [IGNORE_INDEX] * len(source)
        position += len(source)
        wanted = [IGNORE_INDEX] * len(target) if message.get("turn_mask") else target
        assert labels[position : position + len(target)] == wanted
        position += len(target)
    assert position == len(ids)


@pytest.mark.parametrize("tool_format", [None, "qwen3_8"])
@pytest.mark.parametrize("whitespace", ["", " \n"])
@pytest.mark.parametrize("preserve_thinking", [False, True])
@pytest.mark.parametrize("thought", ["Inspect it.", ""])
def test_tool_roundtrip_matches_tokenizer(tokenizer, tool_format, whitespace, preserve_thinking, thought):
    tool = {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
    call = {"name": "terminal", "arguments": {"command": whitespace + "pwd" + whitespace}}
    thought = whitespace + thought + whitespace
    observation = whitespace + "/work" + whitespace
    official = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi", "reasoning_content": "Earlier reasoning."},
        {"role": "user", "content": "Show the working directory."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": thought,
            "tool_calls": [{"type": "function", "function": call}],
        },
        {"role": "tool", "content": observation},
        {"role": "assistant", "content": "/work", "reasoning_content": "Use the result."},
    ]
    lf_messages = [
        official[0],
        {"role": "assistant", "content": "<think>Earlier reasoning.</think>Hi"},
        official[2],
        {
            "role": "function",
            "content": ("<think>\n" + thought + "\n</think>\n\n" if thought.strip() else "") + json.dumps(call),
        },
        {"role": "observation", "content": observation},
        {"role": "assistant", "content": "<think>\nUse the result.\n</think>\n\n/work"},
    ]
    expected = tokenizer.apply_chat_template(
        official,
        tools=[tool],
        tokenize=True,
        add_generation_prompt=False,
        reasoning_effort="medium",
        preserve_thinking=preserve_thinking,
    )
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    template = get_template_and_fix_tokenizer(
        tokenizer,
        DataArguments(
            template="qwen3_8", reasoning_effort="medium", tool_format=tool_format, preserve_thinking=preserve_thinking
        ),
    )
    pairs = template.encode_multiturn(
        tokenizer, lf_messages, tools=json.dumps([tool]), discarding_history_cot=not preserve_thinking
    )
    actual = [token for source, target in pairs for token in source + target]
    assert actual == expected
    source, target = template.encode_oneturn(tokenizer, lf_messages, tools=json.dumps([tool]))
    assert source + target == expected
    generation = tokenizer.apply_chat_template(
        official[:-1],
        tools=[tool],
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="medium",
        preserve_thinking=preserve_thinking,
    )
    if hasattr(generation, "keys"):
        generation = generation["input_ids"]
    assert template.encode_generation(tokenizer, lf_messages[:-1], tools=json.dumps([tool])) == generation


@pytest.mark.parametrize("mask_history", [False, True])
def test_sft_history_thinking_removal_requires_mask_history(tokenizer, mask_history):
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "<think>\nEarlier reasoning.\n</think>\n\nFirst answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "<think>\nCurrent reasoning.\n</think>\n\nSecond answer"},
    ]
    args = DataArguments(template="qwen3_8", preserve_thinking=False, mask_history=mask_history)
    template = get_template_and_fix_tokenizer(tokenizer, args)
    obj = SupervisedDatasetProcessor(
        template, tokenizer, AutoProcessor.from_pretrained(tokenizer.name_or_path, local_files_only=True), args
    )
    result = obj._encode_data_example(messages[:-1], messages[-1:], "", "", [], [], [])
    expected = tokenizer.apply_chat_template(
        official_messages(messages),
        tokenize=True,
        add_generation_prompt=False,
        reasoning_effort="xhigh",
        preserve_thinking=not mask_history,
    )
    if hasattr(expected, "keys"):
        expected = expected["input_ids"]
    assert result[0] == expected
    rendered = tokenizer.decode(result[0])
    assert ("Earlier reasoning." in rendered) == (not mask_history)
    assert "Current reasoning." in rendered

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

import asyncio
import importlib
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llamafactory.chat.base_engine import Response
from llamafactory.chat.chat_model import ChatModel
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data.template import TEMPLATES
from llamafactory.hparams import GeneratingArguments


@pytest.fixture
def text_encoder():
    return SimpleNamespace(
        encode=lambda text, **kwargs: list(map(ord, text)),
        convert_tokens_to_ids=lambda token: 0,
        eos_token_id=0,
        pad_token_id=0,
    )


@pytest.mark.parametrize("backend", ["hf", "vllm", "sglang"])
@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_generation_entrypoint(request, monkeypatch, text_encoder, backend, enable_thinking):
    # Exercise engine prompt construction without importing or starting GPU backends.
    monkeypatch.setattr("llamafactory.extras.packages.is_vllm_available", lambda: False)
    monkeypatch.setattr("llamafactory.extras.packages.is_sglang_available", lambda: False)
    module_name = f"llamafactory.chat.{backend}_engine"
    if module_name not in sys.modules:
        request.addfinalizer(lambda: sys.modules.pop(module_name, None))
    module = importlib.import_module(module_name)
    template = deepcopy(TEMPLATES["qwen3_8"])
    template.reasoning_effort = "medium"
    template.enable_thinking = enable_thinking
    template.mm_plugin = get_mm_plugin("base")
    messages = [{"role": "user", "content": "Hi"}]
    generating_args = GeneratingArguments().to_dict()

    if backend == "hf":
        kwargs, length = module.HuggingfaceEngine._process_args(
            SimpleNamespace(device="cpu", config=SimpleNamespace()),
            text_encoder,
            None,
            template,
            generating_args,
            messages,
            input_kwargs={},
        )
        prompt_ids = kwargs["inputs"][0].tolist()
        assert length == len(prompt_ids)
    else:
        engine_type = module.VllmEngine if backend == "vllm" else module.SGLangEngine
        engine = object.__new__(engine_type)
        engine.template, engine.tokenizer, engine.processor = template, text_encoder, None
        engine.generating_args, engine.lora_request = generating_args, None
        if backend == "vllm":
            monkeypatch.setattr(module, "SamplingParams", SimpleNamespace, raising=False)
            engine.model = SimpleNamespace(generate=Mock())
            asyncio.run(engine._generate(messages))
            prompt_ids = engine.model.generate.call_args.args[0]["prompt_token_ids"]
        else:
            engine.base_url = "http://unused.invalid"
            post = Mock(return_value=SimpleNamespace(status_code=200, iter_lines=lambda **kwargs: [b"data: [DONE]"]))
            monkeypatch.setattr(module.requests, "post", post)
            list(asyncio.run(engine._generate(messages)))
            prompt_ids = post.call_args.kwargs["json"]["input_ids"]

    prefix = "<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n"
    assert "".join(map(chr, prompt_ids)) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" + prefix


@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_chat_restores_prefix_for_history(enable_thinking):
    template = deepcopy(TEMPLATES["qwen3_8"])
    template.enable_thinking = enable_thinking
    chunks = ["Check", " the result.\n</think>\n\n", "2"] if enable_thinking is not False else ["2"]

    async def stream_chat(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    model = object.__new__(ChatModel)
    model.engine = SimpleNamespace(
        template=template,
        chat=AsyncMock(return_value=[Response("".join(chunks), 7, 11, "stop")]),
        stream_chat=stream_chat,
    )

    async def collect():
        return "".join([chunk async for chunk in model.astream_chat([])])

    response = asyncio.run(model.achat([]))[0]
    expected = "<think>\nCheck the result.\n</think>\n\n2" if enable_thinking is not False else "2"
    assert response.response_text == asyncio.run(collect()) == expected
    assert template.remove_thought(response.response_text) == "2"
    assert (response.response_length, response.prompt_length) == (7, 11)


@pytest.mark.parametrize("template_name", ["qwen", "qwen3", "glm4_7"])
@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_existing_templates_keep_generation_behavior(text_encoder, template_name, enable_thinking):
    template = deepcopy(TEMPLATES[template_name])
    template.enable_thinking = enable_thinking
    messages = [{"role": "user", "content": "Hi"}]
    prompt, _ = template.encode_oneturn(text_encoder, messages + [{"role": "assistant", "content": ""}])
    assert template.encode_generation(text_encoder, messages) == prompt
    assert template.get_response_prefix() == ""

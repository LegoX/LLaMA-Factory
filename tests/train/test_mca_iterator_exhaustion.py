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

"""CPU regressions for literal media tokens silently ending an MCA epoch."""

import ast
import functools
import math
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate.data_loader import DataLoaderShard
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import Dataset, SequentialSampler
from transformers import PreTrainedTokenizerFast

from llamafactory.data.collator import SFTDataCollatorWith4DAttentionMask
from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor
from llamafactory.hparams import DataArguments


Qwen3_5Model = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5").Qwen3_5Model
ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "src/mcore_adapter/trainer/trainer.py"


def wrap(collator):
    path = ROOT / "src/llamafactory/train/mca/workflow.py"
    function = next(
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "_data_collator_wrapper"
    )
    namespace = dict(
        functools=functools, Sequence=Sequence, Any=Any, deepcopy=deepcopy, torch=torch, IGNORE_INDEX=-100
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name](collator)


class MediaPlugin:
    image_token = "<|image_pad|>"
    audio_token = None

    def process_messages(self, *args):
        raise AssertionError("Frozen MCA vision modules must not inject dummy media into text-only batches")

    def get_mm_inputs(self, images, videos, audios, *args):
        if images:
            return {
                "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
                "pixel_values": torch.ones(len(images) * 16, 1),
            }
        assert not videos and not audios
        return {}


def make_collator(*, padding_side="right", enable_dummy_media=False):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1, "text": 2, "<|image_pad|>": 3, "<|video_pad|>": 4}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        padding_side=padding_side,
    )
    # Use the installed HF implementation of mRoPE without allocating weights.
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        model_type="qwen3_5", image_token_id=3, video_token_id=4, vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    return SFTDataCollatorWith4DAttentionMask(
        tokenizer=tokenizer,
        template=SimpleNamespace(mm_plugin=MediaPlugin()),
        model=model,
        enable_dummy_media=enable_dummy_media,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )


def feature(ids, **kwargs):
    return dict(input_ids=ids, labels=ids.copy(), attention_mask=[1] * len(ids), **kwargs)


@pytest.mark.parametrize("masked", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("cp", [1, 2, 4])
def test_masked_segments_survive_preprocessing_collation_and_packing(monkeypatch, masked, cp):
    monkeypatch.setenv("LF_PACK_SUBSEQ_ALIGN", str(2 * cp))
    collator = make_collator()
    template = SimpleNamespace(
        mm_plugin=SimpleNamespace(
            process_messages=lambda messages, *a: messages,
            process_token_ids=lambda ids, labels, *a: (ids, labels),
        ),
        efficient_eos=False,
        preserve_thinking=True,
        encode_multiturn=lambda *a: [([2] * 4, [2] * 4)],
    )
    processor = PackedSupervisedDatasetProcessor(
        template,
        collator.tokenizer,
        None,
        DataArguments(cutoff_len=24, packing=True, neat_packing=True, turn_mask=True),
    )
    examples = {
        "_prompt": [[dict(role="user", content="q")]] * 2,
        "_response": [[dict(role="assistant", content="a", turn_mask=value)] for value in masked],
        "_system": ["", ""],
        "_tools": ["", ""],
        "_images": [[], []],
        "_videos": [[], []],
        "_audios": [[], []],
    }
    processed = processor.preprocess_dataset(examples)
    batch = wrap(collator)([{key: value[0] for key, value in processed.items()}])
    namespace = dict(
        torch=torch,
        Tensor=torch.Tensor,
        Any=Any,
        mpu=SimpleNamespace(get_context_parallel_world_size=lambda: cp),
        PackedSeqParams=SimpleNamespace,
    )
    for path, names in [
        (ADAPTER.parent / "utils.py", {"get_seqlens_in_batch", "check_pack_seq_aligned"}),
        (ADAPTER, {"_packing_sequence", "_post_compute_loss"}),
    ]:
        methods = [
            n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    trainer = SimpleNamespace(
        args=SimpleNamespace(sequence_packing=True, calculate_per_token_loss=True),
        model=SimpleNamespace(config=SimpleNamespace(position_embedding_type="mrope", context_parallel_size=1)),
        _language_input_names=["input_ids", "labels", "position_ids", "attention_mask"],
    )
    packed = namespace["_packing_sequence"](trainer, batch)
    assert packed["packed_seq_params"].cu_seqlens_q.tolist() == [0, 8, 16]
    assert packed["input_ids"].shape == packed["labels"].shape == packed["loss_weights"].shape == (1, 16)
    torch.testing.assert_close(packed["position_ids"], torch.arange(8).repeat(2).expand(3, 1, -1))
    valid = (packed["labels"] != -100).float()
    loss, tokens, _ = namespace["_post_compute_loss"](trainer, valid, torch.ones_like(valid))
    assert tokens.item() == valid.sum().item() == 4 * masked.count(False)
    assert loss.item() == tokens.item()


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_literal_image_and_video_tokens_remain_text_without_media(padding_side):
    source = [feature([2, 3, 2, 4, 2]), feature([2, 2])]
    original = deepcopy(source)
    batch = wrap(make_collator(padding_side=padding_side))(source)
    assert source == original
    assert "pixel_values" not in batch and "image_grid_thw" not in batch
    assert batch["position_ids"].shape == (3, 2, 8)
    for i, item in enumerate(source):
        valid = batch["attention_mask"][i].bool()
        assert batch["input_ids"][i, valid].tolist() == item["input_ids"][:-1]
        assert batch["labels"][i, valid].tolist() == item["labels"][1:]
        expected = torch.arange(len(item["input_ids"]) - 1).expand(3, -1)
        torch.testing.assert_close(batch["position_ids"][:, i, valid], expected)


@pytest.mark.parametrize("enable_dummy_media", [False, True])
def test_real_image_keeps_multimodal_positions(enable_dummy_media):
    batch = wrap(make_collator(enable_dummy_media=enable_dummy_media))(
        [feature([2, 3, 3, 3, 3, 2, 2], images=["actual-image"])]
    )
    assert "pixel_values" in batch
    expected = torch.tensor([[0, 1, 1, 1, 1, 3], [0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 3]])
    torch.testing.assert_close(batch["position_ids"][:, 0, :6], expected)


def test_bad_media_raises_error_instead_of_silently_exhausting_dataloader():
    # One real image grid cannot serve two separated image-token groups.
    loader = DataLoaderShard(
        [feature([2, 3, 2, 3, 2], images=["actual-image"])],
        batch_size=1,
        collate_fn=wrap(make_collator()),
        num_workers=0,
    )
    with pytest.raises(RuntimeError, match="data-processing error") as exc:
        list(loader)
    assert isinstance(exc.value.__cause__, StopIteration)


def test_accelerate_iterates_all_1000_samples_and_16_padded_steps():
    if not ADAPTER.is_file():
        pytest.skip("Patched MCA sampler checkout unavailable")
    definition = next(
        n for n in ast.parse(ADAPTER.read_text()).body if getattr(n, "name", None) == "PaddedSequentialSampler"
    )
    namespace = dict(SequentialSampler=SequentialSampler, Dataset=Dataset, math=math)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(ADAPTER), "exec"), namespace)
    rows = [feature([i + 10, 2]) for i in range(1000)]
    # This is the position of the real offending swegen trajectory.
    rows[786] = feature([796, 3, 2, 3, 2, 4, 2])
    sampler = namespace[definition.name](rows, 64)
    loader = DataLoaderShard(rows, batch_size=1, sampler=sampler, collate_fn=wrap(make_collator()), num_workers=0)
    for epoch in range(2):
        loader.set_epoch(epoch)
        indices = [int(batch["input_ids"][0, 0]) - 10 for batch in loader]
        assert len(loader) == len(indices) == 1024
        assert len(indices) // 64 == 16
        assert indices[:1000] == list(range(1000))
        assert indices[1000:] == list(range(epoch * 24, (epoch + 1) * 24))


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("literal_first", [False, True])
@pytest.mark.parametrize("modality, token_id", [("image", 3), ("video", 4)])
def test_literal_media_in_mixed_batch_fails_before_embedding_scatter(padding_side, literal_first, modality, token_id):
    real = feature([2, token_id, token_id, token_id, token_id, 2], **{modality + "s": ["real-media"]})
    literal = feature([2, token_id, 2])
    rows = [literal, real] if literal_first else [real, literal]
    with pytest.raises(ValueError, match=f"Literal {modality} tokens in text-only rows"):
        wrap(make_collator(padding_side=padding_side))(rows)


def test_literal_image_with_dummy_media_fails_before_injection():
    with pytest.raises(ValueError, match="Literal image tokens"):
        wrap(make_collator(enable_dummy_media=True))([feature([2, 3, 2])])


def test_plain_text_and_real_image_can_share_batch():
    batch = wrap(make_collator())([feature([2, 2, 2]), feature([2, 3, 3, 3, 3, 2, 2], images=["real-image"])])
    assert batch["position_ids"].shape == (3, 2, 8)
    torch.testing.assert_close(batch["position_ids"][:, 0, :2], torch.arange(2).expand(3, -1))
    expected = torch.tensor([[0, 1, 1, 1, 1, 3], [0, 1, 1, 2, 2, 3], [0, 1, 2, 1, 2, 3]])
    torch.testing.assert_close(batch["position_ids"][:, 1, :6], expected)

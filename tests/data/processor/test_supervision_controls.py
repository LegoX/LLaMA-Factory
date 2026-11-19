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

from types import SimpleNamespace

import pytest

from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
from llamafactory.hparams import DataArguments


def processor(args, efficient_eos=False, packed=False):
    mm = SimpleNamespace(
        process_messages=lambda messages, *a: messages, process_token_ids=lambda ids, labels, *a: (ids, labels)
    )
    template = SimpleNamespace(
        mm_plugin=mm,
        efficient_eos=efficient_eos,
        preserve_thinking=True,
        encode_multiturn=lambda *a: [([10, 11], [12, 13]), ([20, 21], [22, 23])],
    )
    cls = PackedSupervisedDatasetProcessor if packed else SupervisedDatasetProcessor
    return cls(template, SimpleNamespace(eos_token_id=2, pad_token_id=0), None, args)


def example(masks=(True, False)):
    return dict(
        prompt=[
            dict(role="user", content="q1"),
            dict(role="assistant", content="a1", turn_mask=masks[0]),
            dict(role="user", content="q2"),
        ],
        response=[dict(role="assistant", content="a2", turn_mask=masks[1])],
        system="",
        tools="",
        images=[],
        videos=[],
        audios=[],
    )


@pytest.mark.parametrize("masks", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mask_history", [False, True])
def test_efficient_eos_honors_masked_turns(masks, mask_history):
    obj = processor(DataArguments(turn_mask=True, mask_history=mask_history), efficient_eos=True)
    ids, labels, weights, length = obj._encode_data_example(**example(masks))
    assert len(ids) == len(labels) == len(weights) == length == 9
    assert (labels[-1] == -100) == masks[1]
    assert (labels[4] == -100) == (masks[0] or mask_history)
    if all(masks):
        assert set(labels) == {-100} and set(weights) == {0.0}


def test_fractional_turn_weights_and_eos_stay_aligned():
    row = example((False, False))
    row["prompt"][1]["loss_weight"] = 0.25
    row["response"][0]["loss_weight"] = 0.5
    _, labels, weights, _ = processor(DataArguments(), efficient_eos=True)._encode_data_example(**row)
    assert weights == [0, 0, 0.25, 0.25, 0.25, 0, 0.5, 0.5, 0.5]
    assert labels[4] == labels[-1] == 2


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("drop", [False, True])
def test_long_sequence_drop_uses_original_length(packed, drop, monkeypatch):
    monkeypatch.setenv("LF_PACK_SUBSEQ_ALIGN", "4")
    row = example()
    aligned = {"_" + k: [v] for k, v in row.items()}
    obj = processor(DataArguments(cutoff_len=6, drop_long_sequences=drop, packing=packed), packed=packed)
    result = obj.preprocess_dataset(aligned)
    assert len(result.get("input_ids", [])) == (0 if drop else 1)
    for ids, labels, weights in zip(
        result.get("input_ids", []), result.get("labels", []), result.get("loss_weights", [])
    ):
        assert len(ids) == len(labels) == len(weights)


def test_scoring_refuses_implicit_message_weighting():
    row = example((False, False))
    row["response"][0]["loss_weight"] = 0.5
    with pytest.raises(ValueError, match="unweighted"):
        processor(DataArguments(pretrain_score=True))._encode_data_example(**row, score_metadata={})


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("mask_history", [False, True])
def test_empty_final_answer_keeps_only_its_unmasked_eos(masked, mask_history):
    obj = processor(DataArguments(turn_mask=True, mask_history=mask_history), efficient_eos=True)
    obj.template.encode_multiturn = lambda *a: [([10, 11], [12, 13]), ([20, 21], [])]
    result = obj._encode_data_example(**example((False, masked)))
    ids, labels = result[:2]
    assert ids == [10, 11, 12, 13, 20, 21, 2]
    assert labels[-1] == (-100 if masked else 2)
    assert labels[4] == (-100 if mask_history else 2)
    if len(result) == 4:
        assert result[2][-1] == (0.0 if masked else 1.0)

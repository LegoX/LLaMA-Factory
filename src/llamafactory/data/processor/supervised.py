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
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)

MAX_SU_SEQ_IDX = 2**32  # maximum sub-sequence index


def _get_loss_weight(message: dict[str, Any], default: float = 1.0) -> float:
    weight = message.get("loss_weight", default)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        return default

    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Message loss_weight must be finite and nonnegative.")
    return float(weight)


@dataclass
class PackingParams:
    r"""Metadata for a packed sequence: sub-sequence boundaries and multimodal data indices.

    - sequence_boundaries: cumulative token positions, e.g. [0, 100, 250, 512] means 3 sub-seqs
      with token ranges [0,100), [100,250), [250,512). Length = num_sub_seqs + 1.
    - image_subseq_ids / video_subseq_ids / audio_subseq_ids: for each mm item, the 0-based
      sub-sequence index it belongs to. Length = total number of that mm type in the packed sample.
    """

    sequence_boundaries: list[int]
    image_subseq_ids: list[int]
    video_subseq_ids: list[int]
    audio_subseq_ids: list[int]
    right_padding_length: int


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        score_metadata: Optional[dict] = None,
    ) -> tuple[list[int], list[int], list[float], int]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        loss_weights = [0.0] * len(input_ids)
        discarding_history_cot = self.data_args.mask_history and not self.template.preserve_thinking
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools, discarding_history_cot)
        response_loss_weights = [_get_loss_weight(messages[i + 1]) for i in range(0, len(messages), 2)]
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        full_length = total_length + sum(len(source_ids) + len(target_ids) for source_ids, target_ids in encoded_pairs)
        if score_metadata is not None:
            if self.template.efficient_eos or images or videos or audios:
                raise ValueError(
                    "pretrain_score currently requires text-only data and a template without efficient_eos."
                )
            score_metadata.update(
                full_length=full_length,
                turns=[
                    {
                        "turn_id": idx,
                        "start": None,
                        "end": None,
                        "original_target_tokens": len(target),
                        "target_truncated": True,
                        "context_truncated": False,
                    }
                    for idx, (_, target) in enumerate(encoded_pairs)
                ],
            )
        if score_metadata is not None and any(weight != 1.0 for weight in response_loss_weights):
            raise ValueError("pretrain_score requires unweighted assistant turns.")
        context_truncated = False
        turn_masks = []
        if self.data_args.turn_mask:
            turn_masks = [message.get("turn_mask", False) for message in messages[1::2]]
            if len(turn_masks) != len(encoded_pairs):
                raise ValueError(
                    f"The number of assistant turns ({len(turn_masks)}) does not match the number of encoded turns "
                    f"({len(encoded_pairs)})."
                )

        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns
            response_loss_weights = response_loss_weights[::-1]
            if self.data_args.turn_mask:
                turn_masks = turn_masks[::-1]

        final_eos_weight = 0.0
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            if score_metadata is not None:
                context_truncated = context_truncated or source_len < len(source_ids)
                score_metadata["turns"][turn_idx].update(
                    start=len(input_ids) + source_len,
                    end=len(input_ids) + source_len + target_len,
                    target_truncated=target_len < len(target_ids),
                    context_truncated=context_truncated,
                )
                context_truncated = context_truncated or target_len < len(target_ids)
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0 and source_len:
                # This EOS closes the preceding assistant turn, so honor its mask too.
                previous_weight = response_loss_weights[turn_idx - 1]
                previous_masked = self.data_args.mask_history or (
                    self.data_args.turn_mask and turn_masks[turn_idx - 1]
                )
                eos_label = (
                    self.tokenizer.eos_token_id if previous_weight > 0 and not previous_masked else IGNORE_INDEX
                )
                source_label = [eos_label] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            target_weight = response_loss_weights[turn_idx]
            target_masked = (
                (self.data_args.mask_history and turn_idx != 0)
                or (self.data_args.turn_mask and turn_masks[turn_idx])
                or target_weight == 0.0
            )
            target_label = [IGNORE_INDEX] * target_len if target_masked else target_ids

            source_loss_weight = [1.0 if label != IGNORE_INDEX else 0.0 for label in source_label]
            target_loss_weight = [target_weight if label != IGNORE_INDEX else 0.0 for label in target_label]
            if self.template.efficient_eos and turn_idx and source_len and not self.data_args.train_on_prompt:
                source_loss_weight[0] = previous_weight if source_label[0] != IGNORE_INDEX else 0.0
            if turn_idx == 0 or not self.data_args.mask_history:
                # An empty, unmasked answer still supervises its closing EOS.
                final_eos_weight = 0.0 if target_masked else target_weight

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
                loss_weights = source_loss_weight + target_loss_weight + loss_weights
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label
                loss_weights += source_loss_weight + target_loss_weight

        if self.template.efficient_eos:
            eos_weight = final_eos_weight
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id if eos_weight > 0.0 else IGNORE_INDEX]
            loss_weights += [eos_weight if labels[-1] != IGNORE_INDEX else 0.0]

        return input_ids, labels, loss_weights, full_length

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                if self.data_args.pretrain_score:
                    raise ValueError(
                        f"Cannot score invalid SFT example: {examples['_score_source'][i]}. "
                        "Fix its conversation format before scoring."
                    )
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            score_metadata = {} if self.data_args.pretrain_score else None
            input_ids, labels, loss_weights, full_length = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                score_metadata=score_metadata,
            )
            if self.data_args.drop_long_sequences and full_length > self.data_args.cutoff_len:
                logger.warning_rank0(
                    f"Dropped lengthy example with full length {full_length} > {self.data_args.cutoff_len}."
                )
                continue

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["loss_weights"].append(loss_weights)
            if score_metadata is not None:
                score_metadata.update(json.loads(examples["_score_source"][i]))
                score_metadata.update(retained_length=len(input_ids), truncated=full_length > len(input_ids))
                model_inputs["score_metadata"].append(json.dumps(score_metadata, ensure_ascii=False))
            if self.data_args.trajectory_loss_weight:
                model_inputs["trajectory_loss_weight"].append(examples["_loss_weight"][i])
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_loss_weights = [], [], []
        batch_images, batch_videos, batch_audios = [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels, loss_weights, full_length = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            if self.data_args.drop_long_sequences and full_length > self.data_args.cutoff_len:
                continue

            # Optional: pad each packed sub-sequence to a multiple of LF_PACK_SUBSEQ_ALIGN (e.g. 2 * CP size),
            # required by Megatron THD packing with context parallelism. Padding tokens are ignored by the loss.
            pack_align = int(os.environ.get("LF_PACK_SUBSEQ_ALIGN", "0"))
            if self.data_args.neat_packing and pack_align > 1 and len(input_ids) % pack_align:
                pad = pack_align - len(input_ids) % pack_align
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad
                labels = labels + [IGNORE_INDEX] * pad
                loss_weights = loss_weights + [0.0] * pad
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_loss_weights.append(loss_weights)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        requires_packing_params = self.data_args.neat_packing
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_loss_weights, packed_images, packed_videos, packed_audios = [], [], [], []
            if requires_packing_params:
                sequence_boundaries = [0]
                image_subseq_ids: list[int] = []
                video_subseq_ids: list[int] = []
                audio_subseq_ids: list[int] = []

            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_loss_weights += batch_loss_weights[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if requires_packing_params:
                    n_img = len(batch_images[index])
                    n_vid = len(batch_videos[index])
                    n_aud = len(batch_audios[index])
                    sequence_boundaries.append(sequence_boundaries[-1] + len(batch_input_ids[index]))
                    image_subseq_ids.extend([i] * n_img)
                    video_subseq_ids.extend([i] * n_vid)
                    audio_subseq_ids.extend([i] * n_aud)

                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                packed_loss_weights += [0.0] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

                if requires_packing_params:
                    sequence_boundaries.append(sequence_boundaries[-1] + pad_length)

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            if requires_packing_params:
                packing_params = PackingParams(
                    sequence_boundaries=sequence_boundaries,
                    image_subseq_ids=image_subseq_ids or [MAX_SU_SEQ_IDX],  # avoid dataset concat error
                    video_subseq_ids=video_subseq_ids or [MAX_SU_SEQ_IDX],
                    audio_subseq_ids=audio_subseq_ids or [MAX_SU_SEQ_IDX],
                    right_padding_length=pad_length,
                )
                model_inputs["packing_params"].append(asdict(packing_params))

            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["loss_weights"].append(packed_loss_weights)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs

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

from typing import TYPE_CHECKING

from ...extras import logging
from ...extras.constants import AttentionFunction
from ...extras.packages import is_torch_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _flash_attention_forward_with_optional_s_aux(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    s_aux=None,
    **kwargs,
):
    r"""Compatibility wrapper for Transformers versions that unconditionally cast ``s_aux``."""
    from transformers.integrations.flash_attention import get_target_dtype
    from transformers.modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask

    if kwargs.get("output_attentions", False):
        logger.warning_rank0_once(
            "Flash Attention does not support `output_attentions=True`. "
            "Please set your attention to `eager` if you want any of these features."
        )

    seq_len = query.shape[2]
    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape with a zero dimension. FlashAttention does not support inputs with dim=0. "
            "Please check your input shapes or use SDPA instead."
        )

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    target_dtype = get_target_dtype(query, module)
    is_causal = is_causal if is_causal is not None else module.is_causal

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=flash_attn_supports_top_left_mask(),
        target_dtype=target_dtype,
        attn_implementation=module.config._attn_implementation,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        s_aux=s_aux.to(query.dtype) if s_aux is not None else None,
        **kwargs,
    )
    return attn_output, None


def patch_flash_attention_2_s_aux_none() -> None:
    r"""Patch Transformers FA2 wrapper when it crashes on ``s_aux=None``.

    Transformers main already guards this value. Some released versions still call
    ``s_aux.to(...)`` unconditionally, which breaks Qwen3.5/Qwen3.6 MoE/VL FA2 paths
    that do not use attention sinks.
    """
    import inspect

    try:
        from transformers.integrations.flash_attention import flash_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return

    if getattr(flash_attention_forward, "_llamafactory_s_aux_none_patched", False):
        return

    try:
        source = inspect.getsource(flash_attention_forward)
    except (OSError, TypeError):
        source = ""

    if "s_aux=s_aux.to(query.dtype)" not in source:
        return

    setattr(_flash_attention_forward_with_optional_s_aux, "_llamafactory_s_aux_none_patched", True)
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", _flash_attention_forward_with_optional_s_aux)
    logger.info_rank0("Patched Transformers FlashAttention-2 wrapper to allow `s_aux=None`.")


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    from transformers.utils import is_flash_attn_2_available

    if getattr(config, "model_type", None) == "gpt_oss":
        from transformers.integrations.hub_kernels import load_and_register_kernel

        flash_attn3_kernel = "kernels-community/vllm-flash-attn3"
        load_and_register_kernel(flash_attn3_kernel)
        setattr(config, "_attn_implementation", flash_attn3_kernel)
        setattr(config, "_attn_implementation_internal", flash_attn3_kernel)
        model_args.flash_attn = AttentionFunction.FA3

        logger.info_rank0("Using FlashAttention-3 with attention sink for the gpt-oss model.")
        return

    if getattr(config, "model_type", None) == "gemma2":
        if model_args.flash_attn == AttentionFunction.AUTO or model_args.flash_attn == AttentionFunction.FA2:
            if is_flash_attn_2_available():
                if model_args.flash_attn != AttentionFunction.FA2:
                    logger.warning_rank0("Gemma 2 should use flash attention 2, change `flash_attn` to fa2.")
                    model_args.flash_attn = AttentionFunction.FA2
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, use eager attention.")
                model_args.flash_attn = AttentionFunction.DISABLED
        elif model_args.flash_attn == AttentionFunction.SDPA:
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while the SDPA attention does not support it."
            )

    if getattr(config, "model_type", None) in ["youtu", "youtu_vl"]:
        if model_args.flash_attn in (AttentionFunction.AUTO, AttentionFunction.SDPA):
            logger.warning_rank0("Youtu-VL does not support SDPA, forcing eager attention.")
            model_args.flash_attn = AttentionFunction.DISABLED

    if model_args.flash_attn == AttentionFunction.AUTO:
        return

    elif model_args.flash_attn == AttentionFunction.DISABLED:
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == AttentionFunction.SDPA:
        if not is_torch_version_greater_than("2.1.1"):
            logger.warning_rank0("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        from transformers import is_torch_npu_available

        if not (is_flash_attn_2_available() or is_torch_npu_available()):
            logger.warning_rank0("FlashAttention-2 is not installed.")
            return

        patch_flash_attention_2_s_aux_none()
        requested_attn_implementation = "flash_attention_2"
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "kimi_vl":
        setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "youtu_vl":
        setattr(config, "attn_implementation", requested_attn_implementation)
        setattr(config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "vision_config"):
            setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    else:
        logger.info_rank0("Using vanilla attention implementation.")

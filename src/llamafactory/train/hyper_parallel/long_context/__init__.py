"""Opt-in memory controls for unpacked, text-only Qwen3.8 SFT."""
import os
from .chunked_attention_ops import install_chunked_attention_ops
from .memory import patch_chunked_ce
from .segmented_gdn import install_segmented_gdn
from .chunked_mlp import install_chunked_mlp
from .chunked_norm import install_chunked_norm
from .nested_checkpoint import install_nested_checkpoint
from .recompute_offload import install_recompute_offload
from .grouped_checkpoint_candidate import install_grouped_checkpoint


def install(model, group_size=None):
    if model.config.model_type != "qwen3_5":
        raise ValueError("Long-context adapter requires the Qwen3.5/3.8 architecture")
    settings = {"TEXT_ONLY":"1", "FP32_CE":"1", "SHARD_LOSS_HEAD":"1", "CE_SYNC":"1",
        "CE_EMPTY_CACHE":"1", "NESTED_CHECKPOINT":"1", "RECOMPUTE_OFFLOAD":"1",
        "NORM_ELEMENT_BUDGET":"20971520", "GROUP_REENTRANT":"1"}
    for key,value in settings.items():
        os.environ["QWEN38_"+key]=value
    patch_chunked_ce(model)
    install_segmented_gdn(model,16384)
    install_chunked_mlp(model,4096)
    install_chunked_norm(model,4096)
    install_chunked_attention_ops(model,4096)
    install_nested_checkpoint(model)
    install_recompute_offload(model)
    if group_size is None:
        group_size = int(os.getenv("QWEN38_CHECKPOINT_GROUP_SIZE", "4"))
    if group_size not in (1, 2, 4, 8):
        raise ValueError("Checkpoint group size must be 1, 2, 4, or 8")
    install_grouped_checkpoint(model,group_size)


def install_cp(model, mesh, group_size=4):
    from ..cp_alignment import install_cp_text_loss
    os.environ['QWEN38_ACTIVATION_OFFLOAD']='0'
    os.environ['QWEN38_RECOMPUTE_OFFLOAD']='0'
    os.environ['QWEN38_NESTED_CHECKPOINT']='1'
    os.environ['QWEN38_GROUP_REENTRANT']='1'
    os.environ['QWEN38_NORM_ELEMENT_BUDGET']='20971520'
    os.environ['QWEN38_CE_SYNC']='1'
    os.environ['QWEN38_CE_EMPTY_CACHE']='1'
    install_cp_text_loss(model,mesh)
    install_segmented_gdn(model,16384)
    install_chunked_mlp(model,4096)
    install_chunked_norm(model,4096)
    install_chunked_attention_ops(model,4096)
    install_grouped_checkpoint(model,group_size)

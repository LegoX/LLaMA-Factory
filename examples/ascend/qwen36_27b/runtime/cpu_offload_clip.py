"""FSDP2 global gradient clipping; explicitly sum local shard norms through HCCL."""

import torch
import torch.distributed as dist


@torch.no_grad()
def clip_sharded_grad_norm(parameters, max_norm, device, norm_type=2):
    if norm_type != 2:
        raise ValueError("This alignment path requires L2 clipping")
    gradients = []
    squares = []
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad
        if hasattr(grad, "placements"):
            if len(grad.placements) != 1 or not grad.placements[0].is_shard():
                raise ValueError("Expected one-dimensional FSDP shard placement")
            grad = grad.to_local()
        else:
            raise ValueError("Expected FSDP DTensor gradients")
        if grad.device.type not in ["cpu", "npu"]:
            raise ValueError("Expected CPU or NPU gradients")
        gradients.append(grad)
        squares.append(torch.linalg.vector_norm(grad.float()).square())
    total = (torch.stack(squares).sum() if squares else torch.tensor(0.0)).to(device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    total.sqrt_()
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite global gradient norm")
    coefficient = (
        (float(max_norm) / (total.to(gradients[0].device) + 1e-6)).clamp(max=1.0) if gradients else torch.tensor(1.0)
    )
    torch._foreach_mul_(gradients, coefficient)
    return total

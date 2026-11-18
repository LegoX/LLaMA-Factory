"""Token-chunked SiLU MLP with FP32 projection-weight gradient accumulation."""

import torch
import torch.nn.functional as F


def add_weight_gradient(dy, x, acc):
    if x.device.type == "npu":
        import torch_npu

        torch_npu._npu_matmul_add_fp32(dy, x, acc)
    else:
        acc.addmm_(dy.float().t(), x.float())


class ChunkedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wg, wu, wd, chunk):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = torch.empty_like(flat)
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, len(flat), chunk):
                a = flat[start : start + chunk]
                out[start : start + chunk] = F.linear(F.silu(F.linear(a, wg)) * F.linear(a, wu), wd)
        ctx.save_for_backward(x, wg, wu, wd)
        ctx.chunk = chunk
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x, wg, wu, wd = ctx.saved_tensors
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        dy = dy.reshape_as(x)
        dx = torch.empty_like(x)
        dws = [torch.zeros_like(w, dtype=torch.float32) for w in [wg, wu, wd]]
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, len(x), ctx.chunk):
                a = x[start : start + ctx.chunk]
                d = dy[start : start + ctx.chunk]
                g = F.linear(a, wg)
                u = F.linear(a, wu)
                s = F.silu(g)
                z = s * u
                dz = d.matmul(wd)
                du = dz * s
                dg = torch.ops.aten.silu_backward(dz * u, g)
                dx[start : start + ctx.chunk] = dg.matmul(wg) + du.matmul(wu)
                for grad, inp, acc in [(dg, a, dws[0]), (du, a, dws[1]), (d, z, dws[2])]:
                    add_weight_gradient(grad, inp, acc)
        return dx.reshape(shape), dws[0].to(wg.dtype), dws[1].to(wu.dtype), dws[2].to(wd.dtype), None


def install_chunked_mlp(model, chunk=4096):
    for layer in model.model.language_model.layers:
        patch_mlp(layer.mlp, chunk)


def patch_mlp(mlp, chunk=4096):
    if mlp.config.hidden_act not in ["silu", "swish"]:
        raise ValueError("SiLU MLP only")
    if any(m.bias is not None for m in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]):
        raise ValueError("Biasless MLP only")

    def forward(x):
        return ChunkedMLP.apply(x, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight, chunk)

    mlp.forward = forward

"""Row-chunked Qwen RMSNorm preserving native cast locations."""

import os

import torch
import torch.nn.functional as F


def body(x, w, gate, eps, gated):
    xf = x.float()
    norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if gated:
        return ((w * norm.to(x.dtype)) * F.silu(gate.float())).to(x.dtype)
    return (norm * (1 + w.float())).to(x.dtype)


class ChunkedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, gate, eps, gated, chunk):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        gf = gate.reshape_as(xf) if gate is not None else None
        out = torch.empty_like(xf)
        for start in range(0, len(xf), chunk):
            out[start : start + chunk] = body(
                xf[start : start + chunk], w, gf[start : start + chunk] if gf is not None else None, eps, gated
            )
        ctx.save_for_backward(x, w, gate)
        ctx.eps = eps
        ctx.gated = gated
        ctx.chunk = chunk
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x, w, gate = ctx.saved_tensors
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        dy = dy.reshape_as(xf)
        gf = gate.reshape_as(xf) if gate is not None else None
        dx = torch.empty_like(xf)
        dg = torch.empty_like(gf) if gf is not None else None
        dw = torch.zeros_like(w, dtype=torch.float32)
        for start in range(0, len(xf), ctx.chunk):
            a = xf[start : start + ctx.chunk].detach().requires_grad_()
            d = dy[start : start + ctx.chunk]
            with torch.enable_grad():
                if ctx.gated:
                    g = gf[start : start + ctx.chunk].detach().requires_grad_()
                    out = body(a, w.detach(), g, ctx.eps, True)
                    da, dgate = torch.autograd.grad(out, (a, g), d)
                    dg[start : start + ctx.chunk] = dgate
                else:
                    wf = w.detach().float().requires_grad_()
                    out = body(a, wf, None, ctx.eps, False)
                    da, dweight = torch.autograd.grad(out, (a, wf), d)
                    dw.add_(dweight)
            dx[start : start + ctx.chunk] = da
            if ctx.gated:
                norm = (a.float() * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + ctx.eps)).to(a.dtype)
                dscale = (d.float() * F.silu(g.detach().float())).to(a.dtype)
                dw.add_((dscale * norm).float().sum(0))
        return dx.reshape(shape), dw.to(w.dtype), dg.reshape(shape) if dg is not None else None, None, None, None


def install_chunked_norm(model, chunk=4096):
    for module in model.modules():
        name = module.__class__.__name__
        if name in ["Qwen3_5RMSNorm", "Qwen3_5RMSNormGated"]:
            patch_norm(module, chunk)


def patch_norm(module, chunk=4096):
    budget = int(os.getenv("PROBE_NORM_ELEMENT_BUDGET", "0"))
    if budget:
        chunk = max(1, budget // module.weight.numel())
    gated = module.__class__.__name__ == "Qwen3_5RMSNormGated"
    eps = module.variance_epsilon if gated else module.eps

    def forward(x, gate=None):
        return ChunkedNorm.apply(x, module.weight, gate, eps, gated, chunk)

    module.forward = forward

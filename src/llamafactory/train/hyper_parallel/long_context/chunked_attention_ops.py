"""Bound attention projection and depthwise-convolution workspaces."""
import torch,os
import torch.nn.functional as F
from .chunked_mlp import add_weight_gradient
class Linear(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,w,chunk):
  xf=x.reshape(-1,x.shape[-1]);out=x.new_empty((len(xf),w.shape[0]))
  with torch.autocast(device_type=x.device.type,enabled=False):
   for start in range(0,len(xf),chunk):
    out[start:start+chunk]=F.linear(xf[start:start+chunk],w)
  ctx.save_for_backward(x,w);ctx.chunk=chunk
  return out.reshape(*x.shape[:-1],w.shape[0])
 @staticmethod
 def backward(ctx,dy):
  x,w=ctx.saved_tensors;shape=x.shape;x=x.reshape(-1,x.shape[-1]);dy=dy.reshape(-1,w.shape[0]);dx=torch.empty_like(x);dw=torch.zeros_like(w,dtype=torch.float32)
  with torch.autocast(device_type=x.device.type,enabled=False):
   for start in range(0,len(x),ctx.chunk):
    d=dy[start:start+ctx.chunk];a=x[start:start+ctx.chunk]
    dx[start:start+ctx.chunk]=d.matmul(w);add_weight_gradient(d,a,dw)
    if x.device.type == "npu":
     torch.npu.synchronize()
  return dx.reshape(shape),dw.to(w.dtype),None
class Depthwise(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,w,b,padding,chunk):
  out=x.new_empty((x.shape[0],x.shape[1],x.shape[2]+2*padding-w.shape[-1]+1))
  for start in range(0,x.shape[1],chunk):
   end=min(start+chunk,x.shape[1]);out[:,start:end]=F.conv1d(x[:,start:end],w[start:end],b[start:end] if b is not None else None,padding=padding,groups=end-start)
  ctx.save_for_backward(x,w,b);ctx.padding=padding;ctx.chunk=chunk
  return out
 @staticmethod
 def backward(ctx,dy):
  x,w,b=ctx.saved_tensors;dx=torch.empty_like(x);dw=torch.empty_like(w);db=torch.empty_like(b) if b is not None else None
  for start in range(0,x.shape[1],ctx.chunk):
   end=min(start+ctx.chunk,x.shape[1]);a=x[:,start:end].detach().requires_grad_();weight=w[start:end].detach().requires_grad_()
   with torch.enable_grad():
    bias=b[start:end].detach().requires_grad_() if b is not None else None
    out=F.conv1d(a,weight,bias,padding=ctx.padding,groups=end-start)
    grads=torch.autograd.grad(out,(a,weight,bias) if bias is not None else (a,weight),dy[:,start:end])
   dx[:,start:end]=grads[0];dw[start:end]=grads[1]
   if db is not None:db[start:end]=grads[2]
  return dx,dw,db,None,None

def install_chunked_attention_ops(model,chunk=4096):
 # Native projections are the validated default. Experimental projection
 # chunking produced non-finite gradients in the integrated NPU smoke test.
 for layer in model.model.language_model.layers:
  attn=getattr(layer,'self_attn',getattr(layer,'linear_attn',None))
  for module in attn.modules():
   if isinstance(module,torch.nn.Linear) and os.getenv('QWEN38_ATTN_SKIP_LINEAR', '1')!='1':
    if module.bias is not None:raise ValueError('Only biasless attention projections supported')
    def forward(x,m=module):return Linear.apply(x,m.weight,chunk)
    module.forward=forward
   elif isinstance(module,torch.nn.Conv1d) and os.getenv('QWEN38_ATTN_SKIP_CONV')!='1':
    assert module.groups==module.in_channels==module.out_channels and module.stride==(1,) and module.dilation==(1,)
    def forward(x,m=module):return Depthwise.apply(x,m.weight,m.bias,m.padding[0],256)
    module.forward=forward

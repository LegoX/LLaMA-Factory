"""Recompute a bounded attention module, offloading its saved tensors during backward.
Parameters are explicit autograd inputs, so FSDP receives their returned gradients.
Forward keeps no attention intermediates; only one module's recompute is live.
"""
import functools,torch
from torch.utils._pytree import tree_flatten,tree_unflatten
from torch.nn.utils.stateless import _reparametrize_module

def wrap_recompute_offload(module):
 original=module.forward
 @functools.wraps(original)
 def forward(*args,**kwargs):
  if not torch.is_grad_enabled():return original(*args,**kwargs)
  leaves,spec=tree_flatten((args,kwargs));indices=[i for i,x in enumerate(leaves) if isinstance(x,torch.Tensor)]
  tensors=[leaves[i] for i in indices]
  leaves=[None if isinstance(x,torch.Tensor) else x for x in leaves]
  names,params=zip(*module.named_parameters());count=len(params);output_meta={}
  class Recompute(torch.autograd.Function):
   @staticmethod
   def forward(ctx,*inputs):
    ctx.save_for_backward(*inputs);ctx.device=inputs[0].device.type
    ctx.amp=torch.is_autocast_enabled(ctx.device);ctx.amp_dtype=torch.get_autocast_dtype(ctx.device)
    call_leaves=list(leaves)
    for i,value in zip(indices,inputs[count:]):call_leaves[i]=value
    call_args,call_kwargs=tree_unflatten(call_leaves,spec)
    out=original(*call_args,**call_kwargs);flat,outspec=tree_flatten(out)
    output_meta.update(leaves=[None if isinstance(x,torch.Tensor) else x for x in flat],spec=outspec,indices=[i for i,x in enumerate(flat) if isinstance(x,torch.Tensor)])
    return tuple(flat[i] for i in output_meta['indices'])
   @staticmethod
   def backward(ctx,*doutputs):
    inputs=[x.detach().requires_grad_(need) for x,need in zip(ctx.saved_tensors,ctx.needs_input_grad)]
    new_leaves=list(leaves)
    for i,value in zip(indices,inputs[count:]):new_leaves[i]=value
    call_args,call_kwargs=tree_unflatten(new_leaves,spec)
    with torch.enable_grad(),torch.autocast(device_type=ctx.device,enabled=ctx.amp,dtype=ctx.amp_dtype),_reparametrize_module(module,dict(zip(names,inputs[:count]))),torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="npu"):
     out=original(*call_args,**call_kwargs);flat,_=tree_flatten(out)
     outputs=[flat[i] for i in output_meta['indices']]
     required=[x for x in inputs if x.requires_grad]
     gradients=torch.autograd.grad(outputs,required,doutputs,allow_unused=True)
    it=iter(gradients)
    return tuple(next(it) if x.requires_grad else None for x in inputs)
  output=Recompute.apply(*params,*tensors)
  if not isinstance(output,tuple):output=(output,)
  flat=list(output_meta['leaves'])
  for i,value in zip(output_meta['indices'],output):flat[i]=value
  return tree_unflatten(flat,output_meta['spec'])
 module.forward=forward

def install_recompute_offload(model):
 for layer in model.model.language_model.layers:
  wrap_recompute_offload(getattr(layer,'self_attn',getattr(layer,'linear_attn',None)))

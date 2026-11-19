"""Bound Ascend GDN kernel grids while carrying differentiable recurrent state."""
import torch,os,inspect
from .memory import checkpoint

def segmented_rule(base,segment_tokens=16384):
 if segment_tokens%64:raise ValueError('GDN segment size must align to native 64-token chunks')
 parameters=inspect.signature(base).parameters
 def invoke(**kwargs):
  if "query" in parameters:
   for source,target in [("q","query"),("k","key"),("v","value")]:kwargs[target]=kwargs.pop(source)
   if kwargs.pop("scale",None) is not None:raise ValueError("Torch GDN reference uses its native scale")
  return base(**kwargs)
 def run(q,k,v,g,beta,scale=None,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,cu_seqlens=None,chunk_size=64,head_first=False):
  if cu_seqlens is not None or head_first:raise ValueError('This alignment path requires unpacked, time-first sequences')
  if chunk_size!=64:raise ValueError('Expected native GDN chunk size 64')
  state=initial_state;outputs=[]
  for start in range(0,q.shape[1],segment_tokens):
   end=start+segment_tokens
   operation=invoke
   if os.getenv('QWEN38_NESTED_CHECKPOINT')=='1' and torch.is_grad_enabled():
    def operation(**kwargs):return checkpoint(invoke,use_reentrant=False,**kwargs)
   out,state=operation(q=q[:,start:end],k=k[:,start:end],v=v[:,start:end],g=g[:,start:end],beta=beta[:,start:end],scale=scale,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,chunk_size=chunk_size)
   outputs.append(out)
  return torch.cat(outputs,dim=1),state if output_final_state else None
 return run

def install_segmented_gdn(model,segment_tokens=16384):
 count=0
 for layer in model.model.language_model.layers:
  if hasattr(layer,'linear_attn'):
   layer.linear_attn.chunk_gated_delta_rule=segmented_rule(layer.linear_attn.chunk_gated_delta_rule,segment_tokens)
   count+=1
 return count

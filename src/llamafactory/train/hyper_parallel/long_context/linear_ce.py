"""Chunked linear CE with FP32 output-weight gradient accumulation.
The LM head belongs to the FSDP root; inputs/parameters use its BF16 compute dtype.
Only the final complete weight gradient is cast back to the input weight dtype.
"""
import torch,os,json,time,pathlib
import torch.nn.functional as F

class LinearCE(torch.autograd.Function):
 @staticmethod
 def forward(ctx,hidden,weight,labels,chunk_size):
  if hidden.shape[0]!=1:raise ValueError('Alignment CE requires microbatch=1')
  h=hidden[0,:-1];y=labels[0,1:];selected=torch.nonzero(y!=-100).flatten()
  count=selected.numel()
  if not count:raise ValueError('No supervised tokens')
  loss=torch.zeros((),device=hidden.device,dtype=torch.float32)
  with torch.autocast(device_type=hidden.device.type,enabled=False):
   for ids in selected.split(chunk_size):
    logits=F.linear(h.index_select(0,ids),weight)
    loss.add_(F.cross_entropy(logits.float(),y.index_select(0,ids),reduction='sum'))
  ctx.save_for_backward(hidden,weight,labels,selected);ctx.chunk_size=chunk_size
  return loss/count
 @staticmethod
 def backward(ctx,grad_output):
  hidden,weight,labels,selected=ctx.saved_tensors
  bounded=os.getenv("QWEN38_CE_SYNC")=="1" and hidden.device.type=="npu"
  def memory_record(phase):
   directory=os.getenv("QWEN38_CE_TRACE_DIR")
   if directory and hidden.device.type=="npu":
    free,total=torch.npu.mem_get_info()
    record={"time":time.time(),"phase":phase,"length":hidden.shape[1],"allocated":torch.npu.memory_allocated(),"reserved":torch.npu.memory_reserved(),"free":free,"total":total}
    with (pathlib.Path(directory)/("ce-rank"+os.getenv("RANK","0")+".jsonl")).open("a") as f:f.write(json.dumps(record)+"\n")
  if bounded:torch.npu.synchronize()
  memory_record("backward_start")
  if os.getenv("QWEN38_CE_EMPTY_CACHE")=="1" and hidden.device.type=="npu":
   # ATB workspace allocations need physical free memory outside the torch cache.
   torch.npu.synchronize()
   torch.npu.empty_cache()
   memory_record("backward_cache_released")
  h=hidden[0,:-1];y=labels[0,1:]
  dh=torch.zeros_like(hidden);dw=torch.zeros_like(weight,dtype=torch.float32)
  with torch.autocast(device_type=hidden.device.type,enabled=False):
   for ids in selected.split(ctx.chunk_size):
    x=h.index_select(0,ids);target=y.index_select(0,ids)
    # Reuse the native CE backward to preserve its FP32 softmax/rounding behavior.
    with torch.enable_grad():
     logits=F.linear(x,weight).float().detach().requires_grad_(True)
     loss=F.cross_entropy(logits,target,reduction='sum')
     dl=torch.autograd.grad(loss,logits,grad_outputs=grad_output/selected.numel())[0].to(weight.dtype)
    dh[0,:-1].index_copy_(0,ids,dl.matmul(weight).to(hidden.dtype))
    if hidden.device.type=='npu':
     import torch_npu
     # ATB computes C += x.T @ weight directly into an FP32 accumulator.
     torch_npu._npu_matmul_add_fp32(dl,x,dw)
    else:dw.addmm_(dl.float().t(),x.float())
    if bounded:
     # Bound outstanding ATB workspace lifetimes without changing arithmetic.
     torch.npu.synchronize()
    del x,target,logits,loss,dl
  memory_record("backward_end")
  return dh,dw.to(weight.dtype),None,None

def linear_ce(hidden,labels,head,chunk_size=256):
 if head.bias is not None:raise ValueError('This Qwen LM head must have no bias')
 return LinearCE.apply(hidden,head.weight,labels,chunk_size)

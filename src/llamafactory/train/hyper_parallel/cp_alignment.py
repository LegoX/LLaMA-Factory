"""Text-only CP alignment: preserve causal targets and sample-mean scaling."""
import functools
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast
from .long_context.memory import ChunkedLossHead
from .long_context.linear_ce import LinearCE


def local_causal_targets(labels, local_length, cp_rank):
    shifted=F.pad(labels[:,1:],(0,1),value=-100)
    return shifted[:,cp_rank*local_length:(cp_rank+1)*local_length]


class CPChunkedLossHead(ChunkedLossHead):
    def forward(self, hidden, labels=None):
        if labels is None:
            return F.linear(hidden,self.weight,self.bias)
        size=self._cp_mesh.size();rank=self._cp_mesh.get_local_rank()
        assert hidden.shape[0]==1 and labels.shape[1]==hidden.shape[1]*size
        targets=local_causal_targets(labels,hidden.shape[1],rank)
        count=targets.ne(-100).sum()
        denominator=labels[:,1:].ne(-100).sum().clamp_min(1)
        if count.item():
            # Reuse the validated CE primitive, with an extra trailing hidden
            # position so its causal shift includes the cross-shard target.
            expanded=F.pad(hidden,(0,0,0,1))
            expanded_labels=F.pad(targets,(1,0),value=-100)
            local_mean=LinearCE.apply(expanded,self.weight,expanded_labels,256)
            local_sum=local_mean*count
        else:
            local_sum=(hidden.sum()+self.weight.sum())*0.0
        local_sum=local_sum.float()  # All ranks must reduce the same dtype, including empty shards.
        backward_loss=local_sum*size/denominator
        logging_sum=local_sum.detach().clone()
        dist.all_reduce(logging_sum,group=self._cp_mesh.get_group())
        logging_loss=logging_sum/denominator
        return backward_loss+(logging_loss-backward_loss).detach()


def prepare_cp_inputs(inputs,mesh):
    inputs=dict(inputs)
    mask=inputs['attention_mask'];assert mask.ndim==2 and mask.shape[0]==1
    length=int(mask[0].sum())
    assert bool(mask[0,:length].all()) and not bool(mask[0,length:].any())
    assert not bool((inputs['labels'][:,length:]!=-100).any())
    size=mesh.size();rank=mesh.get_local_rank()
    padded=((length+size*64-1)//(size*64))*(size*64)
    ids=F.pad(inputs['input_ids'][:,:length],(0,padded-length),value=0)
    labels=F.pad(inputs['labels'][:,:length],(0,padded-length),value=-100)
    local=padded//size;start=rank*local
    positions=torch.arange(start,start+local,device=ids.device).unsqueeze(0)
    return {'input_ids':ids[:,start:start+local], 'labels':labels,
            'attention_mask':None,'position_ids':positions,'use_cache':False}


def install_cp_text_loss(model,mesh):
    assert not model.config.tie_word_embeddings
    model.lm_head.__class__=CPChunkedLossHead
    model.lm_head._cp_mesh=mesh
    original=model.forward
    @functools.wraps(original)
    def forward(input_ids=None,attention_mask=None,position_ids=None,labels=None,**kwargs):
        if labels is None:
            return original(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids,**kwargs)
        assert attention_mask is None
        assert not bool(((input_ids==model.config.image_token_id)|(input_ids==model.config.video_token_id)).any())
        for key in ('num_items_in_batch','logits_to_keep','return_dict'):
            kwargs.pop(key,None)
        output=model.model.language_model(input_ids=input_ids,attention_mask=None,position_ids=position_ids,**kwargs)
        loss=model.lm_head(output[0],labels)
        return Qwen3_5CausalLMOutputWithPast(loss=loss,logits=None)
    model.forward=forward

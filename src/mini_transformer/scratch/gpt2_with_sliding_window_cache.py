"""
This script is used to compare the difference between GPT2 with kv cache and without kv cache. This script is a complete implementation of GPT2 with optimized kv cache.
"""

import torch
import torch.nn as nn
from math import sqrt
from ..models.gpt2 import (
    GPT2Config,
    GPT2LayerNorm,
    GPT2FeedForward,
    GPT2PreTrainedModel,
)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, kv_window_size=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        tmp = torch.ones(context_length, context_length)
        self.register_buffer("mask", torch.triu(tmp, diagonal=1))
        
        # Initialize kv_cache buffers
        self.kv_window_size = kv_window_size or context_length
        self.register_buffer('cache_k', None, persistent=False)
        self.register_buffer('cache_v', None, persistent=False)
        self.ptr = 0

    def forward(self, x, use_cache=False):
        batch_size, seq_len_q, _embed_dim = x.shape

        # The shape of self.W_q(x) is (batch_size, sequence_length, d_out) where
        # d_out = num_heads * head_dim. We reshape it here.
        q = self.W_q(x).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, seq_len_q, self.num_heads, self.head_dim)

        # Swap the sequence_length and num_heads dimensions so that we can use matmul.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        d_k = k.shape[-1]  # head_dim here
        
        if use_cache:
            if self.cache_k is None or self.cache_k.shape[0] != batch_size:
                self.cache_k = torch.zeros(
                    batch_size, self.num_heads, self.kv_window_size,
                    self.head_dim, device=x.device
                )
                self.cache_v = torch.zeros_like(self.cache_k)
            
            # discard oldest tokens if incoming chunk overflow
            if self.ptr + seq_len_q > self.kv_window_size:
                overflow = self.ptr + seq_len_q - self.kv_window_size
                # shift everything left by `overflow` (cheap view-copy)
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone() # type: ignore
                
            self.cache_k[:, :, self.ptr:self.ptr+seq_len_q, :] = k
            self.cache_v[:, :, self.ptr:self.ptr+seq_len_q, :] = v # type: ignore
            self.ptr += seq_len_q
            
            k = self.cache_k[:, :, :self.ptr, :]
            v = self.cache_v[:, :, :self.ptr, :] # type: ignore

        attn_scores = q @ k.transpose(2, 3)
        seq_len_k = k.shape[2]
        
        if seq_len_q == seq_len_k:
            # No cache -> use the pre-baked triangular mask slice
            mask = self.mask.bool()[:seq_len_q, :seq_len_k]  # type: ignore
        else:
            # Cached: offset the diagonal by seq_len_k - seq_len
            mask = self.mask.bool()[self.ptr:self.ptr+seq_len_q, :seq_len_k]  # type: ignore

        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / sqrt(d_k), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = (attn_weights @ v).transpose(1, 2) # Swap back num_heads and sequence_length

        # Merge multiple heads into one dimension. Since context_vecs has undergone
        # transpose operation, it is necessary to first use contiguous to ensure that
        # it is continuous in memory.
        context_vecs = context_vecs.contiguous().view(
            batch_size, seq_len_q, self.d_out
        )
        context_vecs = self.out_proj(context_vecs)  # Apply output projection

        return context_vecs
        
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr = 0
        
        
class GPT2TransformerBlock(nn.Module):
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = config.embd_dim,
            d_out = config.embd_dim,
            context_length= config.context_length,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            qkv_bias=config.qkv_bias,
            kv_window_size=config.kv_window_size
        )
        self.ffn = GPT2FeedForward(config)
        self.norm1 = GPT2LayerNorm(config)
        self.norm2 = GPT2LayerNorm(config)
        self.drop_shortcut = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x
        
        
class GPT2Model(GPT2PreTrainedModel):
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        
        self.tok_embd = nn.Embedding(config.vocab_size, config.embd_dim)
        self.pos_embd = nn.Embedding(config.context_length, config.embd_dim)
        self.drop_embd = nn.Dropout(config.embd_pdrop)
        self.trf_blocks = nn.ModuleList(
            [GPT2TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ptr = 0
        self.final_norm = GPT2LayerNorm(config)
    
    def forward(self, input_ids, use_cache=False):
        bsz, seq_len = input_ids.shape
        tok_embds = self.tok_embd(input_ids)
        
        if use_cache:
            pos_ids = torch.arange(self.ptr, self.ptr+seq_len, device=input_ids.device)
            self.ptr += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=input_ids.device)
        pos_embds = self.pos_embd(pos_ids)
            
        x = tok_embds + pos_embds
        x = self.drop_embd(x)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        outputs = self.final_norm(x)
        
        return outputs
        
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.attn.reset_cache() # type: ignore
        self.ptr = 0
        
        
class GPT2ModelForCausalLM(GPT2PreTrainedModel):
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.model = GPT2Model(config)
        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size, bias=False)
        
    def forward(self, input_ids, use_cache=False):
        model_outputs = self.model(input_ids, use_cache=use_cache)
        logits = self.lm_head(model_outputs)
        
        return logits
        
    def reset_kv_cache(self):
        self.model.reset_kv_cache()
        

def generate_text_with_kv_cache(model, input_ids, max_new_tokens, context_length) -> torch.Tensor:
     model.eval()
     
     with torch.no_grad():
         # Init cache with full prompt
         model.reset_kv_cache()
         context_ids = input_ids[:, -context_length:]
         logits = model(context_ids, use_cache=True)
         
         for _ in range(max_new_tokens):
             logits = logits[:, -1, :]
             idx_next = torch.argmax(logits, dim=-1, keepdim=True)
             input_ids = torch.cat((input_ids, idx_next), dim=1)
             
             # This is the key point! We only pass the new token to the model!
             logits = model(idx_next, use_cache=True) 
     
     return input_ids


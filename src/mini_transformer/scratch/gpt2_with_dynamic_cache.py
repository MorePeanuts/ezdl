"""
This script is used to compare the difference between GPT2 with kv cache and without kv cache. This script is a complete implementation of GPT2 with kv cache.
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
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        tmp = torch.ones(context_length, context_length)
        self.register_buffer('mask', torch.triu(tmp, diagonal=1))

        # Initialize kv_cache buffers
        self.register_buffer('cache_k', None, persistent=False)
        self.register_buffer('cache_v', None, persistent=False)
        self.ptr = 0

    def forward(self, x, use_cache=False):
        batch_size, sequence_length, _embed_dim = x.shape

        # The shape of self.W_q(x) is (batch_size, sequence_length, d_out) where
        # d_out = num_heads * head_dim. We reshape it here.
        q = self.W_q(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Concatenate k and v along the dimension of sequence_length.
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)  # type: ignore
            k, v = self.cache_k, self.cache_v

        # Swap the sequence_length and num_heads dimensions so that we can use matmul.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        d_k = k.shape[-1]  # head_dim here

        # attn_scores has dimension (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_scores = q @ k.transpose(2, 3)

        # seq_len_q is the sequence length of the current query,
        # while seq_len_k is the total length of the historical sequence.
        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]
        if use_cache:
            mask = self.mask.bool()[self.ptr : self.ptr + seq_len_q, 0:seq_len_k]  # type: ignore
            self.ptr += seq_len_q
        else:
            mask = self.mask.bool()[:seq_len_q, :seq_len_k]  # type: ignore

        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / sqrt(d_k), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = (attn_weights @ v).transpose(1, 2)  # Swap back num_heads and sequence_length

        # Merge multiple heads into one dimension. Since context_vecs has undergone
        # transpose operation, it is necessary to first use contiguous to ensure that
        # it is continuous in memory.
        context_vecs = context_vecs.contiguous().view(batch_size, sequence_length, self.d_out)
        context_vecs = self.out_proj(context_vecs)  # Apply output projection

        return context_vecs

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr = 0


class GPT2TransformerBlock(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=config.embd_dim,
            d_out=config.embd_dim,
            context_length=config.context_length,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            qkv_bias=config.qkv_bias,
        )
        self.ffn = GPT2FeedForward(config)
        self.norm1 = GPT2LayerNorm(config)
        self.norm2 = GPT2LayerNorm(config)
        self.drop_shortcut = nn.Dropout(config.resid_pdrop)

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, use_cache)
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
            pos_ids = torch.arange(self.ptr, self.ptr + seq_len, device=input_ids.device)
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
            blk.attn.reset_cache()  # type: ignore
        self.ptr = 0


class GPT2ModelForCausalLM(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.model = GPT2Model(config)
        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size, bias=False)

    def forward(self, input_ids, use_cache=False):
        model_outputs = self.model(input_ids, use_cache)
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

"""
Lightweight GPT-2 module implementing core building blocks: a custom layer normalization,
feed-forward MLP, transformer block, the decoder-only GPT-2 model, and a causal LM head.
Includes a simple (deprecated) greedy text generation helper for demonstration.
"""

import torch
import torch.nn as nn
from ..modeling_utils import PreTrainedModel
from .configuration_gpt2 import GPT2Config
from ..activation_func import get_activation_function
from ...scratch.self_attention import MultiHeadAttention


class GPT2PreTrainedModel(PreTrainedModel):
    """
    Base class for GPT-2 models providing shared initialization and utility behavior.
    Acts as a thin wrapper around PreTrainedModel.
    """
    def __init__(self, *args):
        super().__init__()


class GPT2LayerNorm(nn.Module):
    """
    GPT-2 style LayerNorm operating over the last dimension with learnable scale and shift.
    Uses an epsilon from the config for numerical stability.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.eps = config.layer_norm_epsilon
        self.scale = nn.Parameter(torch.ones(config.embd_dim))
        self.shift = nn.Parameter(torch.zeros(config.embd_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GPT2FeedForward(nn.Module):
    """
    Position-wise feed-forward network used within GPT-2 blocks:
    Linear -> activation -> Linear mapping hidden features back to embedding size.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        activation_func = get_activation_function(config.activation_func)
        self.layers = nn.Sequential(
            nn.Linear(config.embd_dim, config.n_inner),
            activation_func,
            nn.Linear(config.n_inner, config.embd_dim),
        )

    def forward(self, x):
        return self.layers(x)


class GPT2TransformerBlock(nn.Module):
    """
    Single GPT-2 transformer block with pre-layer normalization, multi-head self-attention,
    residual connections, dropout on residual paths, and a feed-forward MLP.
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = config.embd_dim,
            d_out = config.embd_dim,
            context_length= config.context_length,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            qkv_bias=config.qkv_bias
        )
        self.ffn = GPT2FeedForward(config)
        self.norm1 = GPT2LayerNorm(config)
        self.norm2 = GPT2LayerNorm(config)
        self.drop_shortcut = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPT2Model(GPT2PreTrainedModel):
    """
    Decoder-only GPT-2 backbone: token and position embeddings, a stack of transformer blocks,
    and a final layer normalization. Produces hidden states for each input position.
    """
    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.tok_embd = nn.Embedding(config.vocab_size, config.embd_dim)
        self.pos_embd = nn.Embedding(config.context_length, config.embd_dim)
        self.drop_embd = nn.Dropout(config.embd_pdrop)
        self.trf_blocks = nn.Sequential(
            *[GPT2TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_norm = GPT2LayerNorm(config)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        tok_embds = self.tok_embd(input_ids)
        pos_embds = self.pos_embd(torch.arange(seq_len, device=input_ids.device))
        x = tok_embds + pos_embds
        x = self.drop_embd(x)
        x = self.trf_blocks(x)
        outputs = self.final_norm(x)

        return outputs


class GPT2ModelForCausalLM(GPT2PreTrainedModel):
    """
    GPT-2 model for causal language modeling. Wraps the GPT-2 backbone and applies
    a linear head to produce vocabulary logits at each time step.
    """
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.model = GPT2Model(config)
        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size, bias=False)

    def forward(self, input_ids):
        model_outputs = self.model(input_ids)
        logits = self.lm_head(model_outputs)

        return logits

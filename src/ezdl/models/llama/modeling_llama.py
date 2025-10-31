"""
"""

import torch
import torch.nn as nn
from ..modeling_utils import PreTrainedModel
from .configuration_llama import LlamaConfig
from ..modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from ..activation_func import get_activation_function
from ..generate_utils import GenerationMixin
from ..cache_utils import Cache, DynamicCache


class LlamaPreTrainedModel(PreTrainedModel):
    
    config_class = LlamaConfig
    base_model_prefix = 'transformer'
    
    def __init__(self, config: LlamaConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)


class LlamaRMSNorm(nn.Module):
    
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size)).float()
        
    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return x_normed * self.weight
        
        
class LlamaMLP(nn.Module):
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mlp_bias = config.mlp_bias
        self.gate_proj = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.intermediate_size, 
            bias=self.mlp_bias
        )
        self.up_proj = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.intermediate_size, 
            bias=self.mlp_bias
        )
        self.down_proj = nn.Linear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=self.mlp_bias
        )
        self.act_fn = get_activation_function(config.hidden_act)
        
    def forward(self, x):
        out_gate = self.act_fn(self.gate_proj(x))
        out_up = self.up_proj(x)
        out_down = self.down_proj(out_gate * out_up)
        return out_down
        
        
class LlamaRotaryEmbedding(nn.Module):
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        
    def forward(self, x, position_ids):
        pass


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, n_kv_heads, seqlen, head_dim) to
    (batch, num_attn_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Equivalent to hidden_states.unsqueeze(2).expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups) # type: ignore
    value_states = repeat_kv(value, module.num_key_value_groups) # type: ignore
    key_seq_len = key_states.shape[-2]
    
    attn_weights = (query @ key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # 4D attention_mask shape: (batch_size, 1, query_length, kv_length)
        # shape[1] = 1 is used to broadcast through all heads.
        causal_mask = attention_mask[:, :, :, : key_seq_len]
        attn_weights = attn_weights + causal_mask
        
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.dropout(attn_weights, p=dropout, train=module.training)
    attn_output = attn_weights @ value_states
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output, attn_weights


class LlamaGroupedQueryAttention(nn.Module):
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        assert config.hidden_size % config.num_attention_heads == 0, f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}"
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert config.num_attention_heads % config.num_key_value_heads == 0, f"Number of attention heads {config.num_attention_heads} must be divisible by number of key-value heads {config.num_key_value_heads}"
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _hidden_size = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        
        # q,k,v states shape: (batch_size, num_heads, seq_len, head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    
class LlamaDecoderLayer(nn.Module):
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn = LlamaGroupedQueryAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config=config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self attention layer
        hidden_states, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        hidden_states = residual + hidden_states
        # Fully connected layer
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    
class LlamaModel(LlamaPreTrainedModel):
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # The embedding vector corresponding to index `padding_idx` has a gradient of 0 during backpropagation.
        # When calculating losses, padding positions are usually ignored.
        self.tok_embd = nn.Embedding(config.vocab_size, config.hidden_size, self.pad_token_id)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_embd = LlamaRotaryEmbedding(config)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        inputs_embeds: torch.Tensor = self.tok_embd(input_ids)
        inputs_seq_len = inputs_embeds.shape[1]
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # `cache_position` represents the absolute position index of the current input token in the entire sequence
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_seq_len, device=inputs_embeds.device
            )
            
        if position_ids is None:
            # `position_ids` is used to generate rotary position embeddings, representing the position index of each 
            # token in the current batch
            position_ids = cache_position.unsqueeze(0) # Add batch dimension
            
        causal_mask = create_causal_mask()
        
        # forward propagation
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_embd(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs
            )
        hidden_states = self.norm(hidden_states)
        
        return BaseModelOutputWithPast(
            last_hidden_state = hidden_states,
            past_key_values = past_key_values,
        )
        

class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        # labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        base_output: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_state = base_output.last_hidden_state
        assert hidden_state is not None, "hidden_state should not be None"
        # Only compute necessary logits, and do not upcase them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_state[:, slice_indices, :])
        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=base_output.past_key_values,
        )
        
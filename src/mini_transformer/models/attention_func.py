import torch
import torch.nn as nn


def eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
):
    """
    Compute scaled dot-product attention with optional masking and dropout.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, query_seq_len, head_dim)
        key (torch.Tensor): Key tensor of shape (batch_size, num_kv_heads, key_seq_len, head_dim)
        value (torch.Tensor): Value tensor of shape (batch_size, num_kv_heads, key_seq_len, head_dim)
        attn_mask (torch.Tensor | None): Optional attention mask tensor
        dropout_p (float): Dropout probability
        is_causal (bool): Whether to apply causal masking
        scale (float | None): Scaling factor for attention scores
        enable_gqa (bool): Whether to enable grouped query attention

    Returns:
        tuple: (attention_output, attention_weights)
            - attention_output: Output tensor of shape (batch_size, query_seq_len, num_heads, head_dim)
            - attention_weights: Attention weights tensor of shape (batch_size, num_heads, query_seq_len, key_seq_len)
    """
    _, num_heads, query_seq_len, _ = query.shape
    _, num_kv_heads, key_seq_len, _ = key.shape

    # Handle grouped query attention
    if enable_gqa:
        n_rep = num_heads // num_kv_heads
        key = torch.repeat_interleave(key, n_rep, dim=1)
        value = torch.repeat_interleave(value, n_rep, dim=1)

    # Compute attention scores
    attn_weights = (query @ key.transpose(2, 3))

    # Apply scaling
    if scale is not None:
        attn_weights *= scale

    # Handle causal masking
    if attn_mask is None and is_causal:
        assert query_seq_len == key_seq_len, "Query and key sequence lengths must be equal for causal attention"
        attn_mask = torch.triu(torch.ones(query_seq_len, query_seq_len, device=query.device), diagonal=1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) * (-torch.inf)

    # Apply attention mask
    if attn_mask is not None:
        # Ensure mask has compatible dimensions
        causal_mask = attn_mask[:, :, :, :key_seq_len]
        attn_weights = attn_weights + causal_mask

    # Compute attention probabilities
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # Apply dropout
    attn_weights = torch.dropout(attn_weights, p=dropout_p, train=(dropout_p > 0))

    # Compute attention output
    attn_output = attn_weights @ value
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
    
    
scaled_dot_product_attention = nn.functional.scaled_dot_product_attention

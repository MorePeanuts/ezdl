"""
Various implementations of attention mechanisms for deep learning models.

This module contains multiple implementations of self-attention and multi-head attention
mechanisms, ranging from basic implementations to optimized versions using different
computational approaches. The implementations are primarily inspired by and reference
the code from https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03.

Classes:
    SelfAttention: Basic scaled dot-product self-attention implementation
    CausalAttention: Self-attention with causal masking for autoregressive models
    MultiHeadAttentionWrapper: Wrapper for single-head attention to create multi-head attention
    MultiHeadAttention: Standard multi-head attention implementation
    MultiHeadAttentionCombinedQKV: Multi-head attention with combined QKV projection
    MultiHeadAttentionEinsum: Multi-head attention using Einstein summation for efficiency
    MultiHeadAttentionScaledDotProduct: Multi-head attention with explicit scaled dot-product
    MultiHeadAttentionPytorch: Multi-head attention using PyTorch's built-in operations
    FlashAttentionScratch: Scratch implementation of Flash Attention for memory efficiency

All implementations support configurable dimensions, number of heads, and attention
parameters, making them suitable for experimentation and educational purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class SelfAttention(nn.Module):
    """
    A simple scaled dot-product self-attention module.

    Projects inputs into query (Q), key (K), and value (V) via linear layers,
    computes attention scores with Q @ K^T scaled by sqrt(d_k), applies softmax
    to obtain attention weights, and returns the weighted sum of values.
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        Initializes the SelfAttention module.

        Args:
            d_in (int): Input feature dimension.
            d_out (int): Output feature dimension for Q, K, and V.
            qkv_bias (bool): Whether to include bias in the linear projections.
        """
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        d_k = k.shape[-1]
        attn_scores = q @ k.T
        attn_weights = torch.softmax(attn_scores / sqrt(d_k), dim=-1)
        context_vecs = attn_weights @ v

        return context_vecs


class CausalAttention(nn.Module):
    """
    Causal (masked) scaled dot-product self-attention for autoregressive models.

    This module projects input tokens into query (Q), key (K), and value (V) spaces,
    computes attention scores via Q @ K^T, scales by the square root of the key
    dimension, applies a lower-triangular causal mask so each position can only
    attend to itself and previous positions, and then applies dropout to the
    attention weights. The mask is registered as a buffer sized by context_length
    so it automatically moves with the module across devices.

    Input shape: (batch_size, sequence_length, d_in)
    Output shape: (batch_size, sequence_length, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        Initialize the CausalAttention module.

        Args:
            d_in (int): Input feature dimension per token.
            d_out (int): Output feature dimension for Q, K, and V projections.
            context_length (int): Maximum sequence length; defines the mask size.
            dropout (float): Dropout probability applied to attention weights.
            qkv_bias (bool, optional): Whether to include bias in Q/K/V linear layers. Defaults to False.
        """
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        tmp = torch.ones(context_length, context_length)

        # Buffers are automatically migrated to the appropriate device (CPU or GPU)
        # along with the model - which means we don't need to manually ensure these
        # tensors are on the same device as the model parameters, thus avoiding
        # device mismatch errors.
        self.register_buffer("mask", torch.tril(tmp, diagonal=1))

    def forward(self, x):
        assert x.ndim == 3, "Input must be a 3D tensor"
        batch_size, sequence_length, _embed_dim = x.shape
        k = self.W_k(x)
        q = self.W_q(x)
        v = self.W_v(x)
        d_k = k.shape[-1]

        attn_scores = q @ k.transpose(1, 2)
        mask = self.mask.bool()[:sequence_length, :sequence_length]  # type: ignore
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / sqrt(d_k), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = attn_weights @ v

        return context_vecs


class MultiHeadAttentionWrapper(nn.Module):
    """
    A wrapper class that implements multi-head attention by stacking multiple CausalAttention modules.

    This simple implementation creates multiple independent causal attention heads and concatenates
    their outputs along the feature dimension. Each head operates on the full input dimension and
    produces an output of dimension d_out.
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize the MultiHeadAttentionWrapper.

        Args:
            d_in (int): Input feature dimension per token.
            d_out (int): Output feature dimension for each attention head.
            context_length (int): Maximum sequence length for causal masking.
            dropout (float): Dropout probability applied to attention weights.
            num_heads (int): Number of attention heads to create.
            qkv_bias (bool, optional): Whether to include bias in Q/K/V linear layers. Defaults to False.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        head_dim = d_out // num_heads
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, head_dim, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal scaled dot-product attention.

    This module implements a standard multi-head attention block for autoregressive models.
    It linearly projects the input into queries (Q), keys (K), and values (V), splits the
    projections into multiple heads, computes scaled dot-product attention with an
    upper-triangular causal mask so each position can only attend to itself and previous
    positions, applies dropout to the attention weights, aggregates values, concatenates
    the heads, and applies a final output projection.

    Input shape: (batch_size, sequence_length, d_in)
    Output shape: (batch_size, sequence_length, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize the MultiHeadAttention module.

        Args:
            d_in (int): Input feature dimension per token.
            d_out (int): Total output feature dimension; equals num_heads * head_dim.
            context_length (int): Maximum sequence length; used to build the causal mask buffer.
            dropout (float): Dropout probability applied to attention weights.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): Whether to include bias in Q/K/V linear projections. Defaults to False.
        """
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

    def forward(self, x):
        batch_size, sequence_length, _embed_dim = x.shape

        # The shape of self.W_q(x) is (batch_size, sequence_length, d_out) where
        # d_out = num_heads * head_dim. We reshape it here.
        q = self.W_q(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Swap the sequence_length and num_heads dimensions so that we can use matmul.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        d_k = k.shape[-1]  # head_dim here

        attn_scores = q @ k.transpose(2, 3)
        mask = self.mask.bool()[:sequence_length, :sequence_length]  # type: ignore
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / sqrt(d_k), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = (attn_weights @ v).transpose(1, 2) # Swap back num_heads and sequence_length

        # Merge multiple heads into one dimension. Since context_vecs has undergone
        # transpose operation, it is necessary to first use contiguous to ensure that
        # it is continuous in memory.
        context_vecs = context_vecs.contiguous().view(
            batch_size, sequence_length, self.d_out
        )
        context_vecs = self.out_proj(context_vecs)  # Apply output projection

        return context_vecs


class MultiHeadAttentionCombinedQKV(nn.Module):
    """
    Multi-head causal scaled dot-product attention using a single combined QKV projection.

    This module computes queries (Q), keys (K), and values (V) with one linear layer,
    reshapes them into multiple heads, applies an upper-triangular causal mask so each
    position can only attend to itself and previous positions, applies dropout to the
    attention weights, aggregates the values, concatenates heads, and projects the result
    with a final output projection.

    Input shape: (batch_size, sequence_length, d_in)
    Output shape: (batch_size, sequence_length, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize the MultiHeadAttentionCombinedQKV module.

        Args:
            d_in (int): Input feature dimension per token.
            d_out (int): Total output feature dimension; equals num_heads * head_dim.
            context_length (int): Maximum sequence length; defines the size of the causal mask buffer.
            dropout (float): Dropout probability applied to attention weights.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): Whether to include bias in the combined Q/K/V linear projection. Defaults to False.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.W_qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        tmp = torch.ones(context_length, context_length)
        self.register_buffer("mask", torch.triu(tmp, diagonal=1))

    def forward(self, x):
        batch_size, sequence_length, _embed_dim = x.shape

        # The shape of self.W_qkv(x) is (batch_size, sequence_length, d_out * 3)
        qkv = self.W_qkv(x).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        # After permutation, the shape becomes (3, batch_size, num_heads, sequence_length, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)
        d_k = k.shape[-1]

        attn_scores = q @ k.transpose(-2, -1)
        mask = self.mask.bool()[:sequence_length, :sequence_length]  # type: ignore
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / sqrt(d_k), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Swap num_heads and sequence_length, and then merge multiple heads
        context_vecs = (attn_weights @ v).transpose(1, 2)
        context_vecs = context_vecs.contiguous().view(
            batch_size, sequence_length, self.d_out
        )
        context_vecs = self.out_proj(context_vecs)

        return context_vecs


class MultiHeadAttentionEinsum(nn.Module):
    """
    Multi-head causal scaled dot-product attention implemented with torch.einsum.

    This module computes queries (Q), keys (K), and values (V) using learned parameter
    matrices (without nn.Linear), splits them into multiple heads, applies an upper-triangular
    causal mask so each position can only attend to itself and previous positions, computes
    scaled dot-product attention, applies dropout to the attention weights, aggregates values,
    concatenates the heads, and applies a final output projection.

    Input shape: (batch_size, sequence_length, d_in)
    Output shape: (batch_size, sequence_length, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize the MultiHeadAttentionEinsum module.

        Args:
            d_in (int): Input feature dimension per token.
            d_out (int): Total output feature dimension; equals num_heads * head_dim.
            context_length (int): Maximum sequence length; used to create the causal mask buffer.
            dropout (float): Dropout probability applied to attention weights.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): Whether to include bias terms for Q, K, and V projections. Defaults to False.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.d_out = d_out

        self.W_q = nn.Parameter(torch.randn(d_out, d_in))
        self.W_k = nn.Parameter(torch.randn(d_out, d_in))
        self.W_v = nn.Parameter(torch.randn(d_out, d_in))

        if qkv_bias:
            self.bias_q = nn.Parameter(torch.zeros(d_out))
            self.bias_k = nn.Parameter(torch.zeros(d_out))
            self.bias_v = nn.Parameter(torch.zeros(d_out))
        else:
            self.register_parameter("bias_q", None)
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        tmp = torch.ones(context_length, context_length)
        self.register_buffer("mask", torch.triu(tmp, diagonal=1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W_q, a=sqrt(5))
        nn.init.kaiming_normal_(self.W_k, a=sqrt(5))
        nn.init.kaiming_normal_(self.W_v, a=sqrt(5))
        if self.bias_q is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_q)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias_q, -bound, bound)
            nn.init.uniform_(self.bias_k, -bound, bound)
            nn.init.uniform_(self.bias_v, -bound, bound)

    def forward(self, x):
        b, n, _ = x.shape
        q = torch.einsum("bni,di->bnd", x, self.W_q)
        k = torch.einsum("bni,di->bnd", x, self.W_k)
        v = torch.einsum("bni,di->bnd", x, self.W_v)

        if self.bias_q is not None:
            q += self.bias_q
            k += self.bias_k
            v += self.bias_v

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        d_k = k.shape[-1]

        attn_scores = torch.einsum("bhnd,bhmd->bhnm", q, k)
        mask = self.mask.bool()[:n, :n]  # type: ignore
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / sqrt(d_k), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vecs = torch.einsum("bhnm,bhmd->bhnd", attn_weights, v)
        context_vecs = context_vecs.transpose(1, 2).reshape(b, n, self.d_out)
        context_vecs = self.out_proj(context_vecs)

        return context_vecs


class MultiHeadAttentionScaledDotProduct(nn.Module):
    """
    Multi-head scaled dot-product attention with causal masking.

    This implementation uses torch.nn.functional.scaled_dot_product_attention,
    which selects the most efficient kernel available (e.g., FlashAttention)
    when supported by the current device and input shapes. Queries, keys, and
    values are computed with a single combined linear projection, split into
    multiple heads, attended with causal masking, then concatenated and passed
    through an output projection.

    Input shape: (batch_size, sequence_length, d_in)
    Output shape: (batch_size, sequence_length, d_out)
    """

    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        """
        Initialize the MultiHeadAttentionScaledDotProduct module.

        This class uses the default mask implementation in the nn.functional.scaled_dot_product_attention function, so there is no need to manually provide the context length, nor to calculate the mask in __init__.

        Args:
            d_in (int): Input feature dimension per token.
            d_out (int): Total output feature dimension; must be divisible by num_heads.
            dropout (float): Dropout probability applied inside scaled_dot_product_attention during training.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): Whether to include bias in the combined Q/K/V linear projection. Defaults to False.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        use_dropout = 0.0 if not self.training else self.dropout

        # Use PyTorch's `scaled_dot_product_attention` function to calculate
        # the attention. This function can automatically select the efficient
        # FlashAttention to complete the calculation when conditions are met.
        # When `attn_mask=None` and `is_causal=True`, an upper triangular mask
        # is used at the underlying level to complete the calculation of the
        # causal attention mechanism.
        # When `attn_mask=attn_mask` and `is_causal=False`, a custom mask can
        # be used to implement more complex attention mechanisms.
        context_vecs = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )
        context_vecs = context_vecs.transpose(1, 2)
        context_vecs = context_vecs.contiguous().view(b, n, self.d_out)
        context_vecs = self.out_proj(context_vecs)

        return context_vecs


class MultiHeadAttentionPytorch(nn.Module):
    """
    Causal multi-head attention wrapper using torch.nn.MultiheadAttention.

    Applies an upper-triangular causal mask for autoregressive models,
    optionally returns attention weights, and projects the attended output
    back to embed_dim via a final linear layer. Expects inputs with
    shape (batch_size, sequence_length, embed_dim) and batch_first=True.
    """

    def __init__(
        self,
        embed_dim,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        need_weights=True,
    ):
        """
        Initialize the MultiHeadAttentionPytorch module.

        - If `need_weights=True`, the attention weight matrix will be returned simultaneously, but FlashAttention cannot be used for memory optimization;

        - If `need_weights=False`, it will attempt to use the optimized scaled_dot_product_attention function for computation.

        Args:
            d_in (int): Input dimension.
            d_out (int): Output dimension.
            context_length (int): Length of the context.
            dropout (float): Dropout probability.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to use bias for query, key, and value.
            need_weights (bool): Whether to return attention weights.
        """
        super().__init__()

        self.context_length = context_length
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )

        self.need_weights = need_weights
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        tmp = torch.ones(context_length, context_length)
        self.register_buffer("mask", torch.triu(tmp, diagonal=1))

    def forward(self, x):
        b, n, _ = x.shape

        mask = self.mask.bool()[:n, :n]  # type: ignore
        attn_output, _attn_weights = self.multihead_attn(
            x, x, x, attn_mask=mask, need_weights=self.need_weights
        )
        if self.need_weights:
            self.attn_weights = _attn_weights
        context_vecs = self.out_proj(attn_output)

        return context_vecs


class FlashAttentionScratch(nn.Module):
    """
    Flash Attention implementation from scratch for validation purposes.

    This implementation demonstrates the core idea of Flash Attention: computing
    attention in blocks to avoid materializing the full N×N attention matrix,
    which reduces memory usage from O(N²) to O(N). The algorithm processes
    queries and keys in blocks and maintains running statistics to compute
    the softmax normalization incrementally.

    Note: This is a simplified implementation. Real Flash Attention uses more
    sophisticated techniques like online softmax and fused kernels.

    Input shape: (batch_size, sequence_length, d_in)
    Output shape: (batch_size, sequence_length, d_out)
    """

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        block_size=64,
    ):
        """
        Initialize the FlashAttentionScratch module.

        Args:
            d_in (int): Input feature dimension per token.
            d_out (int): Total output feature dimension; must be divisible by num_heads.
            context_length (int): Maximum sequence length.
            dropout (float): Dropout probability applied to attention weights.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): Whether to include bias in Q/K/V linear projections. Defaults to False.
            block_size (int, optional): Block size for computing attention. Defaults to 64.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.block_size = block_size

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Store context length for mask creation
        self.context_length = context_length

    def _softmax_online(self, scores, max_score, sum_exp):
        """
        Online softmax computation that updates running statistics.

        Args:
            scores: New scores to incorporate
            max_score: Current maximum score
            sum_exp: Current sum of exponentials

        Returns:
            Updated max_score, sum_exp, and probabilities
        """
        new_max = torch.maximum(max_score, scores.max(dim=-1, keepdim=True)[0])
        new_sum_exp = sum_exp * torch.exp(max_score - new_max) + torch.exp(
            scores - new_max
        ).sum(dim=-1, keepdim=True)

        # Handle numerical stability
        new_sum_exp = torch.clamp(new_sum_exp, min=1e-8)

        return new_max, new_sum_exp

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        # Compute Q, K, V projections
        q = self.W_q(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Transpose for attention computation: (batch_size, num_heads, sequence_length, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Initialize output tensor
        output = torch.zeros_like(q)

        # Process attention in blocks to demonstrate Flash Attention concept
        for i in range(0, sequence_length, self.block_size):
            i_end = min(i + self.block_size, sequence_length)
            q_block = q[
                :, :, i:i_end, :
            ]  # (batch_size, num_heads, block_size, head_dim)

            # Initialize running statistics for this block
            block_output = torch.zeros_like(q_block)
            block_max = torch.full(
                (batch_size, self.num_heads, i_end - i, 1),
                -float("inf"),
                device=x.device,
            )
            block_sum = torch.zeros(
                (batch_size, self.num_heads, i_end - i, 1), device=x.device
            )

            # Process keys and values in blocks
            for j in range(0, sequence_length, self.block_size):
                j_end = min(j + self.block_size, sequence_length)
                k_block = k[
                    :, :, j:j_end, :
                ]  # (batch_size, num_heads, block_size, head_dim)
                v_block = v[
                    :, :, j:j_end, :
                ]  # (batch_size, num_heads, block_size, head_dim)

                # Compute attention scores for current block pair
                scores = q_block @ k_block.transpose(
                    -2, -1
                )  # (batch_size, num_heads, block_q, block_k)
                scores = scores / (self.head_dim**0.5)

                # Apply causal mask if needed
                # Create full causal mask for current sequence
                full_mask = torch.triu(
                    torch.ones(sequence_length, sequence_length, device=x.device),
                    diagonal=1,
                )
                # Extract mask for current block: queries i:i_end, keys j:j_end
                block_mask = full_mask[i:i_end, j:j_end]

                if block_mask.any():
                    # Create boolean mask for masked_fill_ with proper dimensions
                    mask_bool = block_mask.bool()
                    mask_bool = (
                        mask_bool.unsqueeze(0)
                        .unsqueeze(0)
                        .expand(batch_size, self.num_heads, -1, -1)
                    )
                    scores.masked_fill_(mask_bool, -float("inf"))

                # Online softmax with running statistics
                new_max, new_sum = self._softmax_online(scores, block_max, block_sum)

                # Compute attention weights
                attn_weights = torch.exp(scores - new_max) / new_sum

                # Apply dropout
                if self.training:
                    attn_weights = self.dropout(attn_weights)

                # Update running statistics
                scale = torch.exp(block_max - new_max) * block_sum / new_sum
                block_output = block_output * scale + attn_weights @ v_block
                block_max = new_max
                block_sum = new_sum

            output[:, :, i:i_end, :] = block_output

        # Transpose back: (batch_size, sequence_length, num_heads, head_dim)
        output = output.transpose(1, 2)

        # Merge heads
        output = output.contiguous().view(batch_size, sequence_length, self.d_out)
        output = self.out_proj(output)

        return output

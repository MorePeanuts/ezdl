"""
"""

from ..modeling_utils import PreTrainedConfig


class LlamaConfig(PreTrainedConfig):
    """
    Configuration container for a Llama style Transformer model.

    This class holds the hyperparameters required to instantiate a Llama-like
    architecture (RMSNorm, SiLU activation MLP, rotary positional embeddings,
    causal self-attention). It is typically provided to a model constructor and
    can be saved/loaded via PreTrainedConfig utilities.

    Parameters:
        vocab_size (int, default 32000):
            Size of the tokenizer vocabulary (number of unique tokens).
        hidden_size (int, default 4096):
            Dimensionality of the model's hidden states and token embeddings. Equivalent
            to embedding dimension (`embd_dim` in `GPT2Model`).
        intermediate_size (int, default 11008):
            Dimensionality of the hidden layer inside the MLP/FFN block
            (the expansion size before projecting back to hidden_size). Equivalent to 
            `n_inner` in `GPT2Model`.
        num_hidden_layers (int, default 32):
            Number of Transformer blocks (depth of the network). Equivalent to `n_layer` 
            in `GPT2Model`.
        num_attention_heads (int, default 32):
            Number of query attention heads per Transformer layer. Similar to `n_head` in
            `GPT2Model`, but only set the number of query heads which can be different
            from `num_key_value_heads`.
        num_key_value_heads (int, default 32):
            Number of key/value heads per Transformer layer. If this value is
            smaller than num_attention_heads, grouped-query attention (GQA) is used.
            If equal, standard multi-head attention (MHA) is used.
        hidden_act (str, default 'silu'):
            Activation function used in the MLP (e.g., 'silu').
        max_position_embeddings (int, default 4096):
            Maximum sequence length supported by rotary position embeddings. Equivalent to
            `context_length` in `GPT2Model`. In Llama1, this value is 2048 while in Llama2
            it is 4096.
        rms_norm_eps (float, default 1e-06):
            Epsilon value used by RMSNorm for numerical stability.
        use_cache (bool, default True):
            Whether to return and reuse past key/value states to speed up
            autoregressive generation.
        rope_params (dict, default {}):
            Optional configuration for Rotary Position Embeddings (RoPE).
            Common keys include:
            - 'rope_theta' or 'base' (float): Base frequency (theta) for rotary angles.
            - 'scaling' (dict): Parameters to scale RoPE for extended context
              (e.g., {'type': 'linear', 'factor': 2.0}).
            - Implementation-specific flags (e.g., 'rope_type').
        attention_bias (bool, default False):
            If True, attention projection layers include bias terms.
        attention_dropout (float, default 0.0):
            Dropout probability applied to attention weights.
        mlp_bias (bool, default False):
            If True, MLP linear layers include bias terms.
    """

    model_type = 'llama'

    def __init(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        hidden_act: str = 'silu',
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-06,
        use_cache: bool = True,
        rope_params: dict = {},
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_params = rope_params
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

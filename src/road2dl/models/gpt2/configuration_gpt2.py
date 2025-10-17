from typing import Literal


class GPT2Config:
    
    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 1024, # n_positions
        embd_dim: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        n_inner: int | None = None, # inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_func: str = 'gelu_scratch',
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        use_cache: bool = True,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_inner = n_inner or 4 * embd_dim
        self.activation_func = activation_func
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.qkv_bias = qkv_bias
        self.use_cache = use_cache

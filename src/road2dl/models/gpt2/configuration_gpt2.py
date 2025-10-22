from ..modeling_utils import PreTrainedConfig


class GPT2Config(PreTrainedConfig):
    """
    Configuration container for a GPT-2 style Transformer model.

    Parameters:
        vocab_size (int):
            The number of tokens in the vocabulary. Determines the size of the token embedding matrix.
        context_length (int):
            The maximum sequence length (n_positions) the model can process. Also dictates the size of positional embeddings and the attention mask dimensions.
        embd_dim (int):
            The dimensionality of token embeddings and hidden states (often referred to as n_embd).
        n_head (int):
            The number of attention heads in each Transformer layer. Each head attends to a subset of the embedding dimensions.
        n_layer (int):
            The number of Transformer blocks (layers) stacked in the model.
        n_inner (int | None):
            The hidden size of the feed-forward (MLP) sublayer inside each Transformer block. If None, it defaults to 4 * embd_dim, following the GPT-2 design.
        activation_func (str):
            The activation function used in the MLP sublayer (e.g., 'gelu', 'relu', 'gelu_scratch').
        resid_pdrop (float):
            Dropout probability applied to residual connections and/or outputs of sublayers (attention and MLP).
        embd_pdrop (float):
            Dropout probability applied to token and positional embeddings.
        attn_pdrop (float):
            Dropout probability applied to attention weights (and possibly attention outputs depending on implementation).
        layer_norm_epsilon (float):
            Epsilon value for LayerNorm to improve numerical stability.
        qkv_bias (bool):
            Whether to include learned bias terms in the query, key, and value linear projections of attention.
        use_cache (bool):
            Whether to enable key/value caching for faster autoregressive generation (useful during inference).
        kv_window_size (int): 
            The size of the window for cached keys/values when using a limited or sliding attention cache, controlling how many past tokens are retained for attention. # TODO When implementing dynamic caching, this parameter should be removed. Only used in scratch/gpt2_with_kv_cache_optimized.py now.
    """
    
    model_type = 'gpt2'
    attribute_map = {
        'hidden_size': 'embd_dim',
        'max_position_embeddings': 'context_length',
        'num_attention_heads': 'n_head',
        'num_hidden_layers': 'n_layer'
    }

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
        kv_window_size: int = 256,
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
        self.kv_window_size = kv_window_size

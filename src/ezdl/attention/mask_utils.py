import torch
from typing import Callable
from ezdl.models.modeling_utils import PreTrainedConfig


def eager_mask():
    ...
    
def flash_attention_mask():
    ...
    
def flex_attention_mask():
    ...
    

def create_causal_mask(
    config: PreTrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: torch.Tensor | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None
):
    """
    Create a standard causal mask based on the attention implementation used (stored in the config)
    If `past_key_values` has an hybrid cache structure, this function will return the mask corresponding
    to one of the "full_attention" layers (to align to what is needed in the `modeling_xxx.py` files).
    
    Args:
        config (`PreTrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    
    
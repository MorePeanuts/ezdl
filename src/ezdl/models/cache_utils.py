import torch
from ezdl.models.modeling_utils import PreTrainedConfig


class Cache:
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0
        
    def update(
        self, 
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        **cache_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return key_states, value_states
    
    
class DynamicCache(Cache):
    
    def __init__(
        self,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        pass
    
    
class StaticCache(Cache):
    ...
    
    
class QuantizedCache(Cache):
    ...
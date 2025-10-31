import torch
from typing import Any
from abc import ABC, abstractmethod
from ezdl.models.modeling_utils import PreTrainedConfig


class CacheLayerMixin(ABC):
    """Abstract class for a single layer's cache."""
    
    def __init__(self):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized: bool = False
        
    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor):
        ...
        
    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
        
    @abstractmethod
    def get_seq_length(self) -> int:
        ...
    
    def reset(self) -> None:
        """Resets the cache values while preserving the objects."""
        if self.is_initialized:
            self.keys.zero_() # type: ignore
            self.values.zero_() # type: ignore
        if hasattr(self, 'cumulative_length'):
            self.cumulative_length = 0
            
    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device)) # type: ignore
            self.values = self.values.index_select(0, beam_idx.to(self.keys.device)) # type: ignore


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
    """
    
    is_sliding = False
    
    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        
        assert isinstance(self.keys, torch.Tensor) and isinstance(self.values, torch.Tensor)
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values
    
    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        if not self.is_initialized or self.keys.numel() == 0: # type: ignore
            return 0
        return self.keys.shape[-2] # type: ignore
        

class DynamicSlidingWindowLayer(CacheLayerMixin):
    """
    """
    
    is_sliding = True
    
    def __init__(self, sliding_window: int):
        pass
        
    def lazy_initialization(self, key_states: torch.Tensor):
        pass
        
    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        cache_kwargs: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return key_states, value_states
        
    def get_seq_length(self) -> int:
        return 0


class Cache:
    """
    A `Cache` is a list of `CacheLayerMixin` objects, one per model layer. It serves as
    a contrainer for the Cache of each layer.
    
    Args:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate` will
            be used.
        layer_class_to_replicate (`type[CacheLayerMixin]`, *optional*):
            Only used if `layers` is omitted (`None`), in which case it will be used as the base class for each layer,
            and the layers will be added lazily as soon as `update` is called with a `layer_idx` greater than the current
            list of layers.
    """
    
    def __init__(
        self,
        layers: list[CacheLayerMixin] | None = None,
        layer_class_to_replicate: type[CacheLayerMixin] | None = None,
    ):
        if layers is not None and layer_class_to_replicate is not None:
            raise ValueError(
                "You can construct a Cache either from a list `layers` of all the predefined `CacheLayer`, or from a "
                "`layer_class_to_replicate`, in which case the Cache will append a new layer corresponding to "
                "`layer_class_to_replicate` for each new call to `update` with an idx not already in the Cache."
            )
        if layers is None and layer_class_to_replicate is None:
            raise ValueError(
                "You should provide exactly one of `layers` or `layer_class_to_replicate` to initialize a Cache."
            )
        self.layers = layers if layers else []
        self.layer_class_to_replicate = layer_class_to_replicate
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0
        
    def update(
        self, 
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Args:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        return keys, values
        
    def get_seq_lenth(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()
    
    
class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
    Else, it will simply lazy init a full cache of DynamicLayer.

    Args:
        config (`PreTrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
    """
    
    def __init__(
        self,
        config: PreTrainedConfig | None = None,
    ):
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_type in layer_types:
                # From a cache point of view, both sliding and chunked are the same in how they should behave and how many
                # states they should return - only the mask changes to make them different at the end!
                if layer_type in ("sliding_attention", "chunked_attention"):
                    assert sliding_window is not None, "Sliding window must be provided for sliding or chunked attention"
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer())

        # If config was not passed, then simply lazy init a full cache of DynamicLayer
        if len(layers) == 0:
            super().__init__(layer_class_to_replicate=DynamicLayer)
        else:
            super().__init__(layers=layers)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)
    
    
class StaticCache(Cache):
    ...
    
    
class QuantizedCache(Cache):
    ...
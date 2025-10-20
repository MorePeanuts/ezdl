import torch
from typing import Literal


class GenerationConfig:
    
    def __init__(
        self,
        # control the length of the output
        max_new_tokens: int = 20,
        stop_strings: str | list[str] | None = None,
        # control the generation strategy used
        do_sample: bool = False,
        num_beams: int = 1,
        # control the cache
        use_cache: bool = True,
        cache_implementation: Literal['dynamic', 'static', 'offloaded', 'offloaded_static', 'quantized'] | None = None,
        cache_config: dict | None = None,
        # manipulate the output logits
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        #  define the output variables of generate
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        # special tokens
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.stop_strings = stop_strings
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.use_cache = use_cache
        self.cache_implementation = cache_implementation
        self.cache_config = cache_config
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class GenerationMixin:
    
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kargs,
    ):
        pass

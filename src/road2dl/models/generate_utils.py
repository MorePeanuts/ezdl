import torch


class GenerationConfig:
    
    def __init__(self, **kwargs):
        self.max_length = kwargs.pop('max_length', 20)
        self.max_new_tokens = kwargs.pop('max_new_tokens', None)
        
        self.top_k = kwargs.pop('top_k', 50)
        self.top_p = kwargs.pop('top_p', 1.0)


class GenerationMixin:
    
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kargs,
    ):
        pass

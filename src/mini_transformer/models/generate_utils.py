import torch
from .configuration_utils import GenerationConfig


class GenerationMixin:
    
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kargs,
    ):
        pass

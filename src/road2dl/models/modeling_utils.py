from abc import ABC, abstractmethod
from typing import Optional
import torch.nn as nn


class PreTrainedModel(nn.Module):
    
    config_class = None
    base_model_prefix = ""
    
    def _init_weights(self, module):
        ...
    
    @classmethod
    def from_pretrained(cls):
        ...
        
    @classmethod
    def from_config(cls, config: Optional['PreTrainedConfig']):
        ...
    

class PreTrainedConfig(ABC):
    
    @classmethod
    def from_config(cls):
        ...
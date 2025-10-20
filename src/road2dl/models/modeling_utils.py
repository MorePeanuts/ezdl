import os
import torch.nn as nn


class PreTrainedConfig:
    
    @classmethod
    def from_pretrained(cls):
        ...
        
    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike):
        ...
        
    def save_pretrained(self, save_directory: str | os.PathLike):
        ...
        
    def update(self, config_dict):
        ...
        
    def __repr__(self):
        return ''


class PreTrainedModel(nn.Module):
    
    config_class = None
    base_model_prefix = ""
    
    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        super().__init__()
        self.config = config
        
    
    def _init_weights(self, module):
        ...
    
    @classmethod
    def from_pretrained(cls):
        ...
    
    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        state_dict: dict | None = None,
        max_shard_size: int | str = '5GB',
        safe_serialization: bool = True,
    ):
        ...


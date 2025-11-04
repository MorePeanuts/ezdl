import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from .configuration_utils import PreTrainedConfig


class PreTrainedModel(nn.Module):
    config_class = None
    base_model_prefix = ""

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        """
        Initialize the weights of the module. This is quite general on purpose, in the spirit of what we usually do.
        """
        std = 0.02

        if isinstance(
            module,
            (
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
            ),
        ):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
        elif (
            isinstance(
                module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            )
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
    
    @classmethod
    def from_default_config(
        cls
    ):
        """
        Create a model instance from the default configuration.
        """
        config = cls.config_class() # type: ignore
        model = cls(config)
        for module in model.modules():
            model._init_weights(module)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_directory: str | os.PathLike,
    ):
        """
        Create a model instance from a pretrained model directory.
        """
        config = cls.config_class.from_pretrained(model_directory) # type: ignore
        model = cls(config)
        
        # Initialize weights and biases
        for module in model.modules():
            model._init_weights(module)
        
        # Load weights and biases from safetensors or torch format
        model_paths = [
            Path(model_directory) / 'model.safetensors',
            Path(model_directory) / 'model.pth',
            Path(model_directory) / 'model.pt',
            Path(model_directory) / 'model.bin'
        ]
        for model_path in model_paths:
            if not model_path.exists():
                continue
            if model_path.suffix == '.safetensors':
                try:
                    from safetensors.torch import load_file
                    model.load_state_dict(load_file(model_path), strict=False)
                except ImportError:
                    raise ImportError("Please install safetensors to load safetensors models.")
            else:
                model.load_state_dict(torch.load(model_path), strict=False)
        
        return model
        

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        state_dict: dict | None = None,
        safe_serialization: bool = True,
    ): 
        """
        Save the model to a directory.
        
        Args:
            save_directory (str | os.PathLike): The directory to save the model to.
            state_dict (dict | None, optional): The state dictionary to save. Defaults to None.
            safe_serialization (bool, optional): Whether to use safetensors. Defaults to True.
        """
        self.config.save_pretrained(save_directory)
        if safe_serialization:
            try:
                from safetensors.torch import save_file
                from safetensors.torch import save_model
                save_path = Path(save_directory) / 'model.safetensors'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if state_dict is None:
                    save_model(self, str(save_path))
                else:
                    save_file(state_dict, save_path)
                return 
            except ImportError:
                print('safetensors is not installed so that model will be saved in PyTorch format.')
                
        if state_dict is None:
            state_dict = self.state_dict()
        save_path = Path(save_directory) / 'model.pth'
        torch.save(state_dict, save_path)

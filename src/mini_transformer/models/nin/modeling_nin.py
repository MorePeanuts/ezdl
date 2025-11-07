import torch
import torch.nn as nn
from ..modeling_utils import PreTrainedModel
from .configuration_nin import NiNConfig


class NiNPreTrainedModel(PreTrainedModel):
    
    config_class = NiNConfig
    base_model_prefix = 'cnn'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear | nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
        else:
            super()._init_weights(module)
            
            
class NiNBlock(nn.Module):
    
    def __init__(self, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)
        
        
class NiNModel(NiNPreTrainedModel):
    
    def __init__(self, config: NiNConfig):
        super().__init__(config)
        
        in_channels, in_size = config.in_features[0], config.in_features[1:]
        
        self.blocks = nn.ModuleList()
        num_blocks = len(config.nin_out_channels)
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(zip(
            config.nin_out_channels, config.nin_kernel_size, config.nin_stride, config.nin_padding
        )):
            if i == num_blocks - 1:
                self.blocks.append(nn.Dropout(config.dropout))
                self.blocks.append(NiNBlock(out_channels, kernel_size, stride, padding))
            else:
                self.blocks.append(NiNBlock(out_channels, kernel_size, stride, padding))
                self.blocks.append(nn.MaxPool2d(
                    config.pool_kernel_size, 
                    config.pool_stride,
                    config.pool_padding
                ))
            
        x = torch.randn(1, in_channels, *in_size)
        for block in self.blocks:
            x = block(x)
            
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
        
        
class NiNModelForClassification(NiNPreTrainedModel):
    def __init__(self, config: NiNConfig):
        super().__init__(config)
        self.model = NiNModel(config)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x
        
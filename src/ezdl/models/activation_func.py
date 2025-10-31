import torch
import torch.nn as nn


def get_activation_function(name) -> nn.Module:
    match name:
        case 'gelu_scratch':
            return GELU()
        case 'gelu':
            return nn.GELU()
        case 'silu':
            return nn.SiLU()
        case 'silu_scratch':
            return SiLU()
        case 'relu':
            return nn.ReLU()
        case 'sigmoid':
            return nn.Sigmoid()
        case 'tanh':
            return nn.Tanh()
        case 'leaky_relu':
            return nn.LeakyReLU()
        case 'elu':
            return nn.ELU()
        case 'selu':
            return nn.SELU()
        case 'softplus':
            return nn.Softplus()
        case 'softsign':
            return nn.Softsign()
        case 'softmax':
            return nn.Softmax(dim=-1)
        case 'log_softmax':
            return nn.LogSoftmax(dim=-1)
        case 'identity':
            return nn.Identity()
        case 'none':
            return nn.Identity()
        case _:
            raise ValueError(f"Unknown activation function: {name}")


class GELU(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        
        
class SiLU(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)
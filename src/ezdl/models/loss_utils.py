import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Literal


class MSELoss(nn.Module):
    
    def __init__(self, reduction: Literal['mean', 'sum', 'none'] = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, input: Tensor, target: Tensor):
        return F.mse_loss(input, target, reduction=self.reduction)
        
    def scratch_forward(self, input: Tensor, target: Tensor):
        loss = (input - target) ** 2 / 2
        match self.reduction:
            case 'mean':
                loss = loss.mean()
            case 'sum':
                loss = loss.sum()
            case 'none':
                pass
            case _:
                raise ValueError(f"Invalid reduction: {self.reduction}")
        return loss
        
        
class CrossEntropyLoss(nn.Module):
    
    def __init__(
        self, 
        weight=None, 
        ignore_index=-100, 
        reduction='mean', 
        label_smoothing=0.0
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, input: Tensor, target: Tensor):
        return F.cross_entropy(
            input, 
            target, 
            weight=self.weight, 
            ignore_index=self.ignore_index, 
            reduction=self.reduction, 
            label_smoothing=self.label_smoothing
        )
        
    def scratch_forward(self, input: Tensor, target: Tensor):
        max_o = torch.max(input, dim=-1, keepdim=True).values
        term2 = torch.log(torch.sum(torch.exp(input - max_o), dim=-1))
        n = input.shape[0]
        loss = term2 + max_o - input[torch.arange(n), target]
        match self.reduction:
            case 'mean':
                loss = loss.mean()
            case 'sum':
                loss = loss.sum()
            case 'none':
                pass
            case _:
                raise ValueError(f"Invalid reduction: {self.reduction}")
        return loss

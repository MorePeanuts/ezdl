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
                raise ValueError(f'Invalid reduction: {self.reduction}')
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
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
            label_smoothing=self.label_smoothing,
        )

    def scratch_forward(self, input: Tensor, target: Tensor):
        input = input[target != self.ignore_index, :]
        target = target[target != self.ignore_index]
        n, c = input.shape

        # Handle empty case
        if n == 0:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        # Create smoothed target distribution
        smooth_target = torch.full_like(input, self.label_smoothing / (c - 1))
        smooth_target[torch.arange(n), target] = 1 - self.label_smoothing

        # Numerically stable log softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
        max_o = torch.max(input, dim=-1, keepdim=True).values
        log_probs = (
            input - max_o - torch.log(torch.sum(torch.exp(input - max_o), dim=-1, keepdim=True))
        )

        # Cross entropy with label smoothing: -Σ smooth_target * log_probs
        loss = -torch.sum(smooth_target * log_probs, dim=-1)

        # Apply class weights if provided
        if self.weight is not None:
            # Create weight matrix matching smooth_target shape
            weight_matrix = self.weight.unsqueeze(0).expand(n, -1)
            loss = torch.sum(smooth_target * weight_matrix * (-log_probs), dim=-1)

        match self.reduction:
            case 'mean':
                loss = loss.mean()
            case 'sum':
                loss = loss.sum()
            case 'none':
                pass
            case _:
                raise ValueError(f'Invalid reduction: {self.reduction}')
        return loss

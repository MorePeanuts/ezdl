import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    
    def __init__(self, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, 1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return x @ self.weight + self.bias

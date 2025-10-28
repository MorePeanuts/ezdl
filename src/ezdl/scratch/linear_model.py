import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    
    def __init__(self, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
    
    def forward(self, x):
        return x @ self.weight + self.bias


class MultiLinearRegression(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return x @ self.weight + self.bias


class NaiveSoftmaxRegression(nn.Module):
    
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return torch.softmax(x @ self.weight + self.bias, dim=-1)
        
    @staticmethod
    def cross_entropy_loss(inputs, targets):
        n = targets.shape[0]
        return -torch.log(inputs[torch.arange(n), targets]).mean()

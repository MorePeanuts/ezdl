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


class MLPForClassification(nn.Module):
    
    def __init__(self, in_features, num_classes, hidden_dim, dropout=0., naive_dropout=False):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_dim)
        self.activate = nn.ReLU()
        if not naive_dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout_rate = dropout
        self.naive_dropout = naive_dropout
        self.l2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        o1 = self.l1(x)
        hidden_state = self.activate(o1)
        if not self.naive_dropout:
            hidden_state = self.dropout(hidden_state)
        else:
            hidden_state = self.naive_dropout(hidden_state, self.dropout_rate)
        output = self.l2(hidden_state)
        return output
        
    @staticmethod
    def naive_dropout(x, p):
        assert 0 <= p <= 1, "Dropout probability must be between 0 and 1"
        if p == 1:
            return torch.zeros_like(x)
        mask = (torch.rand(x.shape) > p).float()
        return x * mask / (1.0 - p)

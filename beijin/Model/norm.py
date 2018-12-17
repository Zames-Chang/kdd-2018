import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(features, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)
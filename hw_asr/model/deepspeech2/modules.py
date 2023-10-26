import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, 
            in_features: int,
            out_features: int,
        ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z: Tensor) -> Tensor:
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        output = (z - mean) / (std + self.eps)
        output = self.gamma * output + self.beta

        return output


class Lookahead(nn.Module):
    def __init__(self, n_features:int, n_context: int=80):
        super().__init__()
        self.n_context = n_context
        self.n_features = n_features
        self.pad_limits = (0, self.n_context - 1)
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.n_context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, pad=self.pad_limits, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        return x
    

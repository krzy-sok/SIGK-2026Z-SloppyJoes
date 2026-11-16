import torch.nn as nn
import torch

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(channels)
        )

        self.bias = nn.Parameter(
            torch.zeros(channels)
        )

        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)

        var = (x - mean).pow(2).mean(
            dim=1, keepdim=True
        )

        x = (x - mean) / torch.sqrt(var + self.eps)

        x = x * self.weight.view(1, -1, 1, 1)
        x = x + self.bias.view(1, -1, 1, 1)

        return x
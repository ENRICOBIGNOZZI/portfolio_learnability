import torch
from torch import nn


class Policy(nn.Module):
    """Shared pointwise score, free linear output, all layers trained."""
    def __init__(self, dimension, depth, width, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers = []
        sizes = [dimension] + [width] * depth + [1]
        for j, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if j < depth:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.double()

    def forward(self, x):
        return self.layers(x).squeeze(-1)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def scale_output(self, scalar):
        with torch.no_grad():
            self.layers[-1].weight.mul_(scalar)
            self.layers[-1].bias.mul_(scalar)

import torch
import torch.nn as nn


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super(AdaptiveGroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x, indexes: torch.Tensor):

        N, C, H, W = x.size()
        out = torch.Tensor(N*C, H, W)
        G = self.num_groups
        assert C % G == 0

        group_size = C // G

        x = x.view(N*C, H, W)
        for i in range(N*C // group_size):
            group = x[indexes[group_size*i: (i+1)*group_size], :, :]
            mean = x.mean()
            var = x.var()
            group_normalized = (group-mean) / (var+self.eps).sqrt()
            out[indexes[group_size*i: (i+1)*group_size], :, :] = group_normalized

        out = out.view(N, C, H, W)
        return out * self.weight + self.bias
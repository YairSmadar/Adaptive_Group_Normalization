import torch
import torch.nn as nn


class VariableGroupNorm(nn.Module):
    def __init__(self, num_groups, eps=1e-5):
        super().__init__()

        self.instance_norms = nn.ModuleList([
            torch.nn.InstanceNorm2d(num_features=1, eps=eps, affine=True)
            for _ in range(num_groups)
        ])

    def forward(self, x, group_sizes):
        N, C, H, W = x.size()
        assert sum(group_sizes) == C, \
            "The sum of group sizes should equal the number of channels."

        output = []
        start = 0
        for i, group_size in enumerate(group_sizes):
            end = start + group_size
            x_group = x[:, start:end, :, :].reshape(N, 1, 1, group_size * H * W)
            x_group_normalized = self.instance_norms[i](x_group)
            output.append(x_group_normalized.view(N, group_size, H, W))
            start = end

        return torch.cat(output, dim=1)


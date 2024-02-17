import torch
import torch.nn as nn


class GroupNormMyImpl(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        """
        Initialize GroupNorm module.

        Parameters:
        - num_features: int, total number of channels in the input tensor.
        - num_groups: int, number of groups to divide the channels into.
        - eps: float, added to the denominator for numerical stability.
        - affine: bool, if True, this module has learnable affine parameters.
        """
        super(GroupNormMyImpl, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        self.num_features = num_features
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the learnable parameters (only if affine is True)."""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass of the GroupNorm.

        Parameters:
        - x: input tensor of shape (N, C, H, W) where N is the batch size, C is the channel size,
             H and W are the spatial dimensions.

        Returns:
        - The normalized tensor.
        """
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0, 'num_features must be divisible by num_groups'

        x = x.view(N, G, C // G, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)

        if self.affine:
            x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)

        return x
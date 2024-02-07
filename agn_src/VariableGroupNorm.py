import torch
import torch.nn as nn


class VariableGroupNorm(torch.nn.Module):
    def __init__(self, num_channels: int, group_sizes: torch.Tensor, eps: float=1e-5):
        super(VariableGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.group_sizes = group_sizes
        # Calculate the number of groups
        self.num_groups = len(group_sizes)

        # Initialize alpha and beta parameters for each group
        self.weight = nn.Parameter(torch.ones(self.num_groups))
        self.bias = nn.Parameter(torch.zeros(self.num_groups))

        # Validate group sizes
        assert sum(group_sizes) == num_channels, "Sum of group sizes must equal the number of channels"

    def forward(self, x):
        N, C, H, W = x.shape
        x_flattened = x.view(N, C, -1)

        # Pre-compute group boundaries
        boundaries = [0] + torch.cumsum(self.group_sizes, 0).tolist()

        normalized_groups = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            x_group = x_flattened[:, start:end, :]

            # Compute mean and variance for the group
            mean = x_group.mean(dim=[1, 2], keepdim=True)
            var = x_group.var(dim=[1, 2], keepdim=True, unbiased=False)

            # Normalize
            x_group_normalized = (x_group - mean) / torch.sqrt(var + self.eps)
            normalized_groups.append(x_group_normalized)

        # Concatenate all normalized groups
        normalized_tensor = torch.cat(normalized_groups, dim=1)

        # Convert self.group_sizes to a tensor directly, specifying the device
        if isinstance(self.group_sizes, list):
            group_sizes_tensor = torch.tensor(self.group_sizes, device=self.bias.device, dtype=torch.long)
        else:
            # If it's already a tensor, just ensure it's on the correct device
            group_sizes_tensor = self.group_sizes.to(self.bias.device)

        # Scale and shift using alphas and betas for each group
        alphas_expanded = self.weight.repeat_interleave(self.group_sizes).view(1, C, 1)
        betas_expanded = self.bias.repeat_interleave(self.group_sizes).view(1, C, 1)
        out = normalized_tensor * alphas_expanded + betas_expanded
        out = out.view(N, C, H, W)

        return out

    def set_group_sizes(self, group_sizes):
        self.group_sizes = group_sizes

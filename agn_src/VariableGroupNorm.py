import torch
import torch.nn as nn


class VariableGroupNorm(torch.nn.Module):
    def __init__(self, num_channels: int, group_sizes: torch.Tensor, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(VariableGroupNorm, self).__init__()

        self.num_channels = num_channels
        self.eps = eps
        self.group_sizes = group_sizes

        self.indexes = None
        self.reverse_indexes = None

        # Calculate the number of groups
        self.num_groups = len(group_sizes)
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.device = self.weight.device
        print(self.device)

        self.reset_parameters()

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
            var = x_group.var(dim=[1, 2], keepdim=True)

            # Normalize
            x_group_normalized = (x_group - mean) / torch.sqrt(var + self.eps)
            normalized_groups.append(x_group_normalized)

        # Concatenate all normalized groups
        normalized_tensor = torch.cat(normalized_groups, dim=1)

        # Scale and shift using weight and bias for each channel
        # Note: Since weight and bias are now per channel, we can directly use them without needing to expand
        weight_expanded = self.weight.view(1, C, 1)
        bias_expanded = self.bias.view(1, C, 1)

        out = normalized_tensor * weight_expanded + bias_expanded
        out = out.view(N, C, H, W)

        return out

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def set_group_sizes(self, group_sizes: torch.Tensor) -> None:
        self.group_sizes = group_sizes

    def set_indexes(self, indexes: torch.Tensor) -> None:
        self.indexes = indexes

    def set_reverse_indexes(self, reverse_indexes: torch.Tensor) -> None:
        self.reverse_indexes = reverse_indexes

    def apply_shuffle_indexes(self) -> None:
        if self.indexes is None:
            raise Exception("The indexes must be assign before apply shuffle indexes")

        internal_indexes_per_image = self.indexes[:self.num_channels]
        indexes_contain_all_channels_in_image = self.verify_tensor_contents(internal_indexes_per_image)

        # in case the reverse_indexes doesn't assign yet
        if self.reverse_indexes is not None:
            internal_reverse_indexes_per_image = self.reverse_indexes[:self.num_channels]
            reverse_indexes_contain_all_channels_in_image = self.verify_tensor_contents(
                internal_reverse_indexes_per_image)
        else:
            internal_reverse_indexes_per_image = torch.arange(0, self.num_channels, dtype=torch.int64)
            reverse_indexes_contain_all_channels_in_image = True

        if indexes_contain_all_channels_in_image and reverse_indexes_contain_all_channels_in_image:

            # First, return the weights and biases to the original order
            self.weight = torch.nn.Parameter(torch.index_select(self.weight, 0, internal_reverse_indexes_per_image)).to(self.device)
            self.bias = torch.nn.Parameter(torch.index_select(self.bias, 0, internal_reverse_indexes_per_image)).to(self.device)

            # Then, reorder again according to the new indexes
            self.weight = torch.nn.Parameter(torch.index_select(self.weight, 0, internal_indexes_per_image)).to(self.device)
            self.bias = torch.nn.Parameter(torch.index_select(self.bias, 0, internal_indexes_per_image)).to(self.device)
        else:
            raise Exception("When using VGN, the re-clustering of the channels must be per image")

    def verify_tensor_contents(self, tensor):
        N = self.num_channels

        # Check length
        if len(tensor) != N:
            return False

        # Check for uniqueness and range
        unique_elements = torch.unique(tensor)
        if len(unique_elements) == N and unique_elements.min() == 0 and unique_elements.max() == N - 1:
            return True
        else:
            return False

# if __name__ == '__main__':
#     # Assume VariableGroupNorm is defined as provided earlier
#
#     # Parameters
#     num_channels = 8
#     group_size = 2
#     num_groups = num_channels // group_size
#     eps = 1e-5
#
#     # Create an instance of standard GroupNorm
#     gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)
#
#     # Create an instance of VariableGroupNorm with equal group sizes
#     group_sizes = torch.full((num_groups,), group_size, dtype=torch.int)
#     vgn = VariableGroupNorm(num_channels=num_channels, group_sizes=group_sizes, eps=eps)
#
#     # Input tensor
#     x = torch.rand(2, num_channels, 4, 4)  # Batch size of 2, height and width of 4
#
#     # Pass the input through both normalization layers
#     gn_output = gn(x)
#     vgn_output = vgn(x)
#
#     # Compare the outputs
#     print("Outputs are equal:", torch.allclose(gn_output, vgn_output))

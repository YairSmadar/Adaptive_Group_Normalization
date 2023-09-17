import torch
import torch.nn as nn


class VariableGroupNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, group_sizes, eps):
        N, C, H, W = x.size()
        x_flattened = x.view(N, C, -1)

        # Validate that the provided group sizes are consistent with the number of channels.
        VariableGroupNormFunction._validate_group_sizes(group_sizes, C)

        # Pre-compute group boundaries
        boundaries = list(zip([0] + group_sizes[:-1].cumsum(0).tolist(), group_sizes.cumsum(0).tolist()))

        # Lists to store normalized tensors and statistics for each group.
        normalized_groups, mus, ivars = [], [], []

        for start, end in boundaries:
            # Extract the channels corresponding to the current group.
            x_group = x_flattened[:, start:end, :]

            # Normalize this group and store its statistics.
            xhat, (mu, ivar) = VariableGroupNormFunction._normalize_group(x_group, eps)

            # Save normalized tensor and statistics for the backward pass.
            normalized_groups.append(xhat)
            mus.append(mu)
            ivars.append(ivar)

        # Concatenate all normalized groups to form the full normalized tensor.
        normalized_tensor = torch.cat(normalized_groups, dim=1)

        # Scale and shift the normalized tensor using weight and bias parameters.
        out = (normalized_tensor * weight.view(1, C, 1) + bias.view(1, C, 1)).view(N, C, H, W)

        # Store less data on ctx
        ctx.save_for_backward(normalized_tensor, weight)
        ctx.intermediate_values = (mus, ivars)
        ctx.boundaries = boundaries
        ctx.eps = eps

        return out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors from the forward pass.
        normalized_tensor, weight = ctx.saved_tensors
        mus, ivars = ctx.intermediate_values
        boundaries = ctx.boundaries

        N, C, H, W = grad_output.size()
        grad_output_flattened = grad_output.view(N, C, -1)
        grad_inputs = []

        # Compute gradient for each group.
        for idx, (start, end) in enumerate(boundaries):
            grad_input_group = VariableGroupNormFunction._compute_group_gradient(
                grad_output_flattened[:, start:end, :],
                normalized_tensor[:, start:end, :],
                weight[start:end],
                mus[idx],
                ivars[idx]
            )
            grad_inputs.append(grad_input_group)

        # Concatenate gradients for all groups to form gradient for the full tensor.
        grad_input_tensor = torch.cat(grad_inputs, dim=1).view(N, C, H, W)

        # Compute gradients for weight and bias parameters
        grad_weight = (grad_output_flattened * normalized_tensor).sum(dim=(0, 2))
        grad_bias = grad_output_flattened.sum(dim=(0, 2))

        return grad_input_tensor, grad_weight, grad_bias, None, None

    @staticmethod
    def _validate_group_sizes(group_sizes, C):
        # Ensure all group sizes are positive.
        assert all(group_size > 0 for group_size in group_sizes), "Group size should be greater than zero."
        # Ensure the sum of all group sizes matches the total number of channels.
        assert sum(group_sizes) == C, "The sum of group sizes should equal the number of channels."

    @staticmethod
    def _normalize_group(x_group, eps):
        # Compute mean and variance for the entire group.
        mu = x_group.mean(dim=[1, 2], keepdim=True)  # Mean over the channel and spatial dimensions
        var = x_group.var(dim=[1, 2], keepdim=True)  # Variance over the channel and spatial dimensions
        # Compute standard deviation and its inverse.
        std = torch.sqrt(var + eps)
        ivar = 1.0 / std
        # Normalize the group using computed statistics.
        xhat = (x_group - mu) * ivar
        return xhat, (mu, ivar)

    @staticmethod
    def _compute_group_gradient(grad_output_group, normalized_group, gamma, mu, ivar):
        # Retrieve total elements in the group (channels * spatial dimensions).
        N, G_channels, G_spatial = grad_output_group.size()
        G = G_channels * G_spatial

        # Compute gradient of the normalized values with respect to the input.
        d_normalized = grad_output_group * gamma.view(1, G_channels, 1)

        # Gradient with respect to variance.
        d_var = (-0.5 * ivar * (d_normalized * (normalized_group - mu)).sum(dim=[1, 2], keepdim=True))

        # Gradient with respect to mean.
        d_mu = (-ivar * d_normalized.sum(dim=[1, 2], keepdim=True)) - 2.0 / G * d_var * (normalized_group - mu).sum(
            dim=[1, 2], keepdim=True)

        # Gradient with respect to input x.
        grad_input = d_normalized * ivar + d_mu / G + 2.0 / G * (normalized_group - mu) * d_var

        return grad_input


class VariableGroupNorm(nn.Module):

    def __init__(self, num_channels, eps=1e-12):
        super(VariableGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, group_sizes):
        return VariableGroupNormFunction.apply(x, self.weight, self.bias, group_sizes, self.eps)

    def extra_repr(self):
        return '{num_channels}, group_sizes={group_sizes}, eps={eps}'.format(
            **self.__dict__)


if __name__ == '__main__':
    from torch.autograd import gradcheck

    # Use double precision, required for gradcheck
    x = torch.randn(10, 20, 5, 5, dtype=torch.double, requires_grad=True).double()
    weight = torch.randn(20, dtype=torch.double, requires_grad=True).double()
    bias = torch.randn(20, dtype=torch.double, requires_grad=True).double()
    group_sizes = torch.tensor([10, 10], dtype=torch.int64)

    # Check gradients
    gradcheck(VariableGroupNormFunction.apply, (x, weight, bias, group_sizes, 1e-5))

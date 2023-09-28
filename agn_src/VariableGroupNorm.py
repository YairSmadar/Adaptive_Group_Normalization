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
        boundaries = list(zip([0] + group_sizes[:-1].cumsum(0).tolist(),
                              group_sizes.cumsum(0).tolist()))

        # Lists to store normalized tensors and statistics for each group.
        normalized_groups, mus, ivars = [], [], []

        for start, end in boundaries:
            # Extract the channels corresponding to the current group.
            x_group = x_flattened[:, start:end, :]

            # Normalize this group and store its statistics.
            xhat, (mu, ivar) = VariableGroupNormFunction._normalize_group(
                x_group, eps)

            # Save normalized tensor and statistics for the backward pass.
            normalized_groups.append(xhat)
            mus.append(mu)
            ivars.append(ivar)

            # Adjust scale and bias according to the calculated mean and inverse variance.
            scale = ivar  # since ivar is 1/std
            adjusted_bias = -mu * scale
            if bias is not None:
                group_bias = bias[start:end].view(1, -1, 1)
                adjusted_bias = -mu * scale + group_bias

            xhat = xhat * (weight[start:end].view(1, -1,
                                                  1) if weight is not None else 1) + adjusted_bias
            normalized_groups[-1] = xhat

        # Concatenate all normalized groups to form the full normalized tensor.
        normalized_tensor = torch.cat(normalized_groups, dim=1)

        out = normalized_tensor.view(N, C, H, W)

        # Store data on ctx for backward pass
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
        zeros_as_var = torch.zeros_like(var)
        # Compute standard deviation and its inverse.
        std = torch.sqrt(torch.max(var, zeros_as_var) + eps)
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

         # Cache this term as it's used multiple times
        term = normalized_group - mu

        # Gradient with respect to variance.
        d_var = (-0.5 * ivar * (d_normalized * term).sum(dim=[1, 2], keepdim=True))

        # Gradient with respect to mean.
        d_mu = (-ivar * d_normalized.sum(dim=[1, 2], keepdim=True)) - 2.0 / G * d_var * term.sum(
            dim=[1, 2], keepdim=True)

        # Gradient with respect to input x.
        grad_input = d_normalized * ivar + d_mu / G + 2.0 / G * term * d_var

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


def wrapped_vgn_forward(*args, **kwargs):
    return VariableGroupNormFunction.apply(*[a.double() if torch.is_tensor(a) and a.dtype != torch.int else a for a in args])




if __name__ == '__main__':
    # Create small double precision inputs
    x = torch.randn(2, 6, 5, 5, dtype=torch.double, requires_grad=True)
    weight = torch.randn(6, dtype=torch.double, requires_grad=True)
    bias = torch.randn(6, dtype=torch.double, requires_grad=True)
    group_sizes = torch.tensor([2, 2, 2],
                               dtype=torch.int)  # example group sizes
    eps = torch.tensor(1e-12, dtype=torch.double, requires_grad=False)

    # Perform the gradient check
    check = torch.autograd.gradcheck(wrapped_vgn_forward, (x, weight, bias, group_sizes, eps))
    print(f"Gradcheck result: {check}")  # If True, gradients are correct.


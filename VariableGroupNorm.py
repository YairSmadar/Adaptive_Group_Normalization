import torch
import torch.nn as nn


class VariableGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, group_sizes, eps):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        # Checking the group sizes
        assert all(group_size > 0 for group_size in
                   group_sizes), "Group size should be greater than zero."
        assert all(group_size <= C for group_size in
                   group_sizes), "Group size should not be greater than " \
                                 "the number of channels."
        assert sum(
            group_sizes) == C, "The sum of group sizes should equal " \
                               "the number of channels."

        means = []
        stds = []
        out = []
        start = 0
        for group_size in group_sizes:
            end = start + group_size
            x_group = x[:, start:end, :]
            mean = x_group.mean([-1, -2], keepdim=True)
            std = x_group.std([-1, -2], keepdim=True)
            x_group = (x_group - mean) / (std + eps)
            out.append(x_group)
            means.append(mean)
            stds.append(std)
            start = end
        out = torch.cat(out, dim=1)

        out = out * weight.view(1, C, 1) + bias.view(1, C, 1)
        out = out.view(N, C, H, W)

        means_tensor = torch.stack(means)
        stds_tensor = torch.stack(stds)

        ctx.save_for_backward(x, out, weight, bias, group_sizes, means_tensor,
                              stds_tensor)
        ctx.eps = eps

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Get the saved tensors and epsilon from the forward pass
        x, out, weight, bias, group_sizes, means, stds = ctx.saved_tensors
        eps = ctx.eps

        # Reshape the input tensor, output tensor, and gradient tensor to (N, C, -1)
        N, C, H, W = grad_output.size()
        grad_output = grad_output.view(N, C, -1)
        x = x.view(N, C, -1)
        out = out.view(N, C, -1)

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # Compute the initial gradient: element-wise multiplication of the output gradient and weights
            grad_x = grad_output * weight.view(1, C, 1)

            # grad_x = dxhat

            start = 0
            for i, group_size in enumerate(group_sizes):
                end = start + group_size

                # Extract the group-specific gradient and input, mean and std from the forward pass
                grad_x_group = grad_x[:, start:end, :]
                x_group = x[:, start:end, :]
                mean = means[i]
                std = stds[i]

                N_GROUP = group_size * H * W

                # Compute gradients for mean and standard deviation
                grad_mean = grad_x_group.sum([-1, -2], keepdim=True)  # dbeta

                # grad_x_group = dgammax
                # grad_output = dout
                # x_group = xhat

                # Part 1: Compute the normalized deviation of each
                # element in the group
                deviation = x_group - mean

                # Part 2: Multiply the deviation by the gradient
                grad_times_deviation = grad_x_group * deviation

                # Part 3: Sum over all elements in the group
                sum_grad_times_deviation = \
                    grad_times_deviation.sum([-1, -2], keepdim=True)

                # Part 4: Compute the scaling factor for the gradient
                # f(x) = 1/sqrt(x), we get f'(x) = -0.5 * x^-1.5
                scaling_factor = (-0.5) * (std + eps).pow(-1.5)

                # Part 5: Compute the gradient with respect to
                # the standard deviation
                grad_std = sum_grad_times_deviation * scaling_factor

                ######################################################

                # Compute gradients for input using the chain rule
                # Part 1: Gradient of the group due to the mean
                grad_due_to_mean = grad_mean / N_GROUP

                # Part 2: Gradient of the group due to the standard deviation
                grad_due_to_std = (
                      x_group - mean) * grad_std * 2 / N_GROUP

                # Part 3: Combine the two parts and normalize
                # by the standard deviation
                grad_x_group_combined = (
                        grad_x_group - grad_due_to_mean - grad_due_to_std) / (
                        std + eps).sqrt()

                # Assign the computed gradient to the correct
                # portion of the full gradient tensor
                grad_x[:, start:end, :] = grad_x_group_combined

                start = end

            # Reshape the computed gradients to match the original input shape
            grad_input = grad_x.view(N, C, H, W)

        # Compute gradients for weight and bias parameters, if required
        if any(ctx.needs_input_grad[1:]):
            grad_weight = (grad_output * out).sum([0, 2],
                                                  keepdim=True).squeeze()
            grad_bias = grad_output.sum([0, 2], keepdim=True).squeeze()

        return grad_input, grad_weight, grad_bias, None, None


class VariableGroupNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(VariableGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, group_sizes):
        return VariableGroupNormFunction.apply(x, self.weight, self.bias,
                                               group_sizes, self.eps)

    def extra_repr(self):
        return '{num_channels}, group_sizes={group_sizes}, eps={eps}'.format(
            **self.__dict__)

import torch
import torch.nn as nn


class VariableGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, group_sizes, eps):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

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
        x, out, weight, bias, group_sizes, means, stds = ctx.saved_tensors
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        grad_output = grad_output.view(N, C, -1)
        x = x.view(N, C, -1)
        out = out.view(N, C, -1)

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * weight.view(1, C, 1)
            start = 0
            for i, group_size in enumerate(group_sizes):
                end = start + group_size
                grad_x_group = grad_x[:, start:end, :]
                x_group = x[:, start:end, :]
                mean = means[i]
                std = stds[i]

                N_GROUP = group_size * H * W
                grad_mean = grad_x_group.sum([-1, -2], keepdim=True)
                grad_std = (grad_x_group * (x_group - mean)).sum([-1, -2], keepdim=True) * (-0.5) * (std+eps).pow(-1.5)
                grad_x[:, start:end, :] = (grad_x_group - grad_mean/N_GROUP - (x_group - mean) * 2 * grad_std/N_GROUP) / (std+eps).sqrt()

                start = end
            grad_input = grad_x.view(N, C, H, W)

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
        return VariableGroupNormFunction.apply(x, self.weight, self.bias, group_sizes, self.eps)

    def extra_repr(self):
        return '{num_channels}, group_sizes={group_sizes}, eps={eps}'.format(**self.__dict__)

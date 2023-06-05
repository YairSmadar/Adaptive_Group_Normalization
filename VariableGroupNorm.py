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

        xhats = []
        xmus = []
        ivars = []
        sqrtvars = []
        vars = []

        start = 0
        for group_size in group_sizes:
            end = start + group_size
            x_group = x[:, start:end, :]
            number_of_values = group_size * W * H

            # step1: calculate mean

            mu = torch.div(1., number_of_values) * x_group.sum(axis=(-1, -2), keepdim=True)

            # step2: subtract mean vector of every trainings example
            xmu = x_group - mu

            # step3: following the lower branch - calculation denominator
            sq = xmu ** 2

            # step4: calculate variance
            var = torch.div(1., number_of_values) * sq.sum(axis=(-1, -2), keepdim=True)

            # step5: add eps for numerical stability, then sqrt
            sqrtvar = torch.sqrt(var + eps)

            # step6: invert sqrtwar
            ivar = torch.div(1., sqrtvar)

            # step7: execute normalization
            xhat = xmu * ivar

            xhats.append(xhat)
            xmus.append(xmu)
            ivars.append(ivar)
            sqrtvars.append(sqrtvar)
            vars.append(var)

            start = end

        xhats_tensor = torch.cat(xhats, dim=1)
        xmus_tensor = torch.cat(xmus, dim=1)
        ivars_tensor = torch.stack(ivars, dim=0)
        sqrtvars_tensor = torch.stack(sqrtvars, dim=0)
        vars_tensor = torch.stack(vars, dim=0)

        # step8: Nor the two transformation steps
        gammax = weight.view(1, C, 1) * xhats_tensor

        # step9
        out = gammax + bias.view(1, C, 1)
        out = out.reshape((N, C, H, W))

        ctx.save_for_backward(xhats_tensor, weight, xmus_tensor,
                              ivars_tensor, sqrtvars_tensor, vars_tensor,
                              group_sizes)
        ctx.eps = eps

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output = dout

        https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        """
        # Get the saved tensors and epsilon from the forward pass
        xhats_tensor, gammas_tensor, xmus_tensor, ivars_tensor, \
        sqrtvars_tensor, vars_tensor, group_sizes = ctx.saved_tensors
        eps = ctx.eps

        # Reshape the input tensor, output tensor,
        # and gradient tensor to (N, C, -1)
        N, C, H, W = grad_output.size()
        grad_output = grad_output.view(N, C, -1)

        dxs = []

        dxs_tensor = dbeta = dgamma = None

        if ctx.needs_input_grad[0]:

            # step9
            grad_outputx = grad_output * gammas_tensor.view(1, C, 1)

            # part of step5
            varPeps = vars_tensor + eps
            start = 0
            for i, group_size in enumerate(group_sizes):
                end = start + group_size

                grad_x_group_g = grad_outputx[:, start:end, :]

                # step8
                dxhat = grad_x_group_g

                # step7
                xmus_tensor_g = xmus_tensor[:, start:end, :]
                dxhatMxmu = dxhat * xmus_tensor_g
                divar = dxhatMxmu.sum(axis=[-1, -2], keepdim=True)

                ivars_tensor_g = ivars_tensor[i, :, :]
                dxmu1 = dxhat * ivars_tensor_g

                # step6
                sqrtvars_tensor_g = sqrtvars_tensor[i, :, :]
                dsqrtvar = torch.div(-1., (sqrtvars_tensor_g ** 2)) * divar

                # step5
                varPeps_g = varPeps[i, :, :]
                dvar = 0.5 * torch.div(1., varPeps_g.sqrt()) * dsqrtvar

                # step4
                dsq = torch.div(1., group_size) * torch.ones((N, group_size, H*W)) * dvar

                # step3
                xmus_tensor_g = xmus_tensor[:, start:end, :]
                dxmu2 = 2 * xmus_tensor_g * dsq

                # step2
                dx1 = (dxmu1 + dxmu2)
                dmu = -1 * dx1.sum(axis=[-1, -2], keepdim=True)

                # step1
                dx2 = torch.div(1., group_size) * torch.ones((N, group_size, H*W)) * dmu

                # step0
                dx = dx1 + dx2
                dxs.append(dx)

                start = end

            dxs_tensor = torch.cat(dxs, dim=1).view(dxs[0].size(0), -1, dxs[0].size(2))

            # Reshape the computed gradients to match the original input shape
            dxs_tensor = dxs_tensor.view(N, C, H, W)

            if any(ctx.needs_input_grad):
                # part of step9
                dbeta = grad_output.sum(axis=[0, 2])

                # part of step8
                grad_outputMxhat = grad_output * xhats_tensor
                dgamma = grad_outputMxhat.sum(axis=[0, 2])

        return dxs_tensor, dgamma, dbeta, None, None


class VariableGroupNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-12):
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

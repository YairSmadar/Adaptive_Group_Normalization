import torch
import torch.autograd


class SimplifiedVariableGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        N, C, H, W = input.size()
        mu = input.mean([2, 3], keepdim=True)
        var = input.var([2, 3], keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)

        x_hat = (input - mu) / std
        out = x_hat * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)

        ctx.save_for_backward(input, weight, bias, x_hat, mu, std)
        ctx.eps = eps

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, x_hat, mu, std = ctx.saved_tensors
        eps = ctx.eps
        N, C, H, W = grad_output.size()

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # Calculate gradients for input
            grad_output_hat = grad_output * weight.view(1, C, 1, 1)
            grad_input = grad_output_hat - grad_output_hat.mean([2, 3], keepdim=True) \
                         - x_hat * (grad_output_hat * x_hat).mean([2, 3], keepdim=True)
            grad_input = grad_input / std

        if ctx.needs_input_grad[1]:
            # Calculate gradients for weight
            grad_weight = (grad_output * x_hat).sum(dim=[0, 2, 3])

        if ctx.needs_input_grad[2]:
            # Calculate gradients for bias
            grad_bias = grad_output.sum(dim=[0, 2, 3])

        return grad_input, grad_weight, grad_bias, None


# Testing function with gradcheck
def test_simplified_vgn():
    torch.manual_seed(42)  # For consistent results

    N, C, H, W = 2, 6, 5, 5
    input = torch.randn(N, C, H, W, dtype=torch.double, requires_grad=True)
    weight = torch.randn(C, dtype=torch.double, requires_grad=True)
    bias = torch.randn(C, dtype=torch.double, requires_grad=True)
    eps = 1e-5

    from torch.autograd.gradcheck import gradcheck

    test = gradcheck(SimplifiedVariableGroupNormFunction.apply, (input, weight, bias, eps), eps=1e-6, atol=1e-4,
                     rtol=1e-3)
    print(f"Gradcheck result: {test}")


if __name__ == "__main__":
    test_simplified_vgn()

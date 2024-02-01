import torch


class ReorderChannelsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_indexes, reverse_indexes):
        # Save the reverse indexes for use in the backward pass
        ctx.save_for_backward(reverse_indexes)

        N, C, H, W = input.size()

        input_reshaped = input.view(-1, W * H)

        # Use torch.index_select for better performance
        input_new_idx = torch.index_select(input_reshaped, 0, forward_indexes[:N*C])

        output = input_new_idx.view(N, C, H, W).clone()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors (reverse_indexes)
        reverse_indexes, = ctx.saved_tensors

        N, C, H, W = grad_output.size()

        grad_output_reshaped = grad_output.view(-1, W * H)

        # Use torch.index_select for better performance
        grad_output_new_idx = torch.index_select(grad_output_reshaped, 0, reverse_indexes[:N * C])

        grad_input = grad_output_new_idx.view(N, C, H, W).clone()

        return grad_input, None, None  # Return None for indexes gradients as they do not require gradients


# Wrapper to easily apply our custom Function
def reorder_channels(input, forward_indexes, reverse_indexes):
    return ReorderChannelsFunction.apply(input, forward_indexes, reverse_indexes)
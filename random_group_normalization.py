from torch import tensor, sort, div, randperm, empty_like, zeros, cat, max, arange, argmax
from torch.nn import Module, GroupNorm
import numpy as np
import global_vars
from agn_utils import getLayerIndex


class RandomGroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int = 32, eps=1e-12):
        super(RandomGroupNorm, self).__init__()
        self.groupNorm = GroupNorm(num_groups, num_channels, eps=eps, affine=True)
        self.indexes = None
        self.reverse_indexes = None
        self.layer_index = getLayerIndex()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_size = int(num_channels / num_groups)

    def forward(self, Conv_input):
        N, C, W, H = Conv_input.shape
        if global_vars.recluster:
            self.recluster(N*C)

        Conv_input_reshaped = Conv_input.reshape(N * C, W * H)
        Conv_input_new_idx = Conv_input_reshaped[self.indexes, :]

        return self.groupNorm(Conv_input_new_idx.reshape(N, C, H, W)).requires_grad_(requires_grad=True)

    def recluster(self, number_of_channels_to_group):
        self.indexes = sort(randperm(number_of_channels_to_group, generator=global_vars.generator))[1]
        self.reverse_indexes = empty_like(self.indexes, device=global_vars.device)
        for ind in range(number_of_channels_to_group):
            self.reverse_indexes[self.indexes[ind]] = tensor(ind, device=global_vars.device)

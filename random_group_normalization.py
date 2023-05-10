import torch
from torch import tensor, sort, randperm, empty_like
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
        self.eval_indexes = None
        self.eval_reverse_indexes = None

    def forward(self, Conv_input):
        N, C, W, H = Conv_input.shape

        # start shuffle at epoch > 0
        if global_vars.args.epoch_start_cluster > global_vars.epoch_num:
            return self.groupNorm(Conv_input)

        if global_vars.recluster:
            self.recluster(Conv_input)

        # in case using shuffle last batch
        if self.indexes is None:
            return self.groupNorm(Conv_input)

        if global_vars.train_mode:
            indexes = self.indexes
            reverse_indexes = self.reverse_indexes
        else:
            indexes = self.eval_indexes
            reverse_indexes = self.eval_reverse_indexes

        Conv_input_reshaped = Conv_input.view(-1, W * H)

        # Use torch.index_select for better performance
        Conv_input_new_idx = torch.index_select(Conv_input_reshaped, 0,
                                                indexes)
        GN_input = Conv_input_new_idx.view(N, C, H, W)

        Conv_input_new_idx_norm = self.groupNorm(GN_input)

        Conv_input_new_idx_norm = Conv_input_new_idx_norm.view(-1, W * H)

        # Use torch.index_select for better performance
        Conv_input_orig_idx_norm = torch.index_select(Conv_input_new_idx_norm,
                                                      0, reverse_indexes)

        ret = Conv_input_orig_idx_norm.view(N, C, H, W).requires_grad_(
            requires_grad=True)

        return ret

    def recluster(self, Conv_input):
        N, C, W, H = Conv_input.shape
        RGN_version = global_vars.args.RGN_version

        # random all channels (N*C)
        if RGN_version == 1:
            self.indexes = sort(randperm(N*C, generator=global_vars.generator))[1].to(
                                            Conv_input.device)
            self.reverse_indexes = torch.argsort(self.indexes).to(
                Conv_input.device)
            print(f"Currently, RGN_version number {RGN_version}"
                  f" is not available!")
            exit(1)

        # random only in the same image, all images the same
        elif RGN_version == 2:
            self.indexes = sort(randperm(C, generator=global_vars.generator))[1].to(
                                            Conv_input.device)
            factors = torch.arange(0, N) * C
            self.indexes = torch.cat([self.indexes] * N)
            self.indexes = \
                (self.indexes.reshape(N, C) + factors.unsqueeze(1)).view(-1)
            self.reverse_indexes = torch.argsort(self.indexes).to(
                Conv_input.device)

            self.eval_indexes = self.indexes
            self.eval_reverse_indexes = self.reverse_indexes

        else:
            print(f"RGN_version number {RGN_version} is not available!")
            exit(1)
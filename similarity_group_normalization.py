import time

from torch import clone, empty_like, tensor, sort, zeros_like, cat, ceil, floor
from torch.nn import Module, GroupNorm
from torch.nn.functional import normalize
import torch
import global_vars
from agn_utils import getLayerIndex
from random_group_normalization import better_places
from k_means_constrained import KMeansConstrained
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class SimilarityGroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int = 32, eps=1e-12):
        super(SimilarityGroupNorm, self).__init__()
        self.groupNorm = GroupNorm(num_groups, num_channels, eps=eps, affine=True)
        self.indexes = None
        self.reverse_indexes = None
        self.layer_index = getLayerIndex()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_size = int(num_channels / num_groups)
        self.eps = eps

        self.before_list = []
        self.after_list = []

        self.batch_layer_index = {}

    def forward(self, Conv_input, batch_num: int = -1):
        N, C, H, W = Conv_input.size()

        # if global_vars.args.plot_std:
        #     group_size = int(C / self.num_groups)
        #     input_no_h_w = Conv_input.reshape(N, C, H * W)
        #     channel_dist = cat([input_no_h_w[i, :, :] for i in range(N)], dim=1)
        #     channel_dist_by_groups = cat([channel_dist[i:i+group_size, :].unsqueeze(0) for i in range(int(C/group_size))], dim=0)
        #     std_before = channel_dist_by_groups.reshape(int(C/group_size), -1).std(dim=1)
        #
        #     self.before_list.append(np.sum(std_before.cpu().detach().numpy()))

        if global_vars.recluster:
            self.recluster(Conv_input)
            self.batch_layer_index[batch_num] = (self.indexes.clone(), self.reverse_indexes.clone())
            # raise Exception("stop")

        Conv_input_reshaped = Conv_input.reshape(N * C, W * H)

        # if batch_num == 155:
        #     print()
        t = time.time()
        if global_vars.args.save_shuff_idxs:
            Conv_input_new_idx = Conv_input_reshaped[self.batch_layer_index[batch_num][0], :]
            Conv_input_new_idx_norm = self.groupNorm(Conv_input_new_idx.reshape(N, C, H, W))
            Conv_input_new_idx_norm = Conv_input_new_idx_norm.reshape(N*C, W*H)
            Conv_input_orig_idx_norm = Conv_input_new_idx_norm[self.batch_layer_index[batch_num][1], :]
            ret = Conv_input_orig_idx_norm.reshape(N, C, H, W).requires_grad_(requires_grad=True)
        else:
            Conv_input_new_idx = Conv_input_reshaped[self.indexes, :]
            Conv_input_new_idx_norm = self.groupNorm(Conv_input_new_idx.reshape(N, C, H, W))
            Conv_input_new_idx_norm = Conv_input_new_idx_norm.reshape(N*C, W*H)
            Conv_input_orig_idx_norm = Conv_input_new_idx_norm[self.reverse_indexes, :]
            ret = Conv_input_orig_idx_norm.reshape(N, C, H, W).requires_grad_(requires_grad=True)

        # print(f"norm time {time.time() - t}")


        # if global_vars.args.plot_std:
        #     input_no_h_w = Conv_input[:, self.indexes, :, :].reshape(N, C, H * W)
        #     channel_dist = cat([input_no_h_w[i, :, :] for i in range(N)], dim=1)
        #     channel_dist_by_groups = cat([channel_dist[i:i+group_size, :].unsqueeze(0) for i in range(int(C/group_size))], dim=0)
        #     std_after = channel_dist_by_groups.reshape(int(C/group_size), -1).std(dim=1)
        #     self.after_list.append(np.sum(std_after.cpu().detach().numpy()))
        #
        #     print("std before - after:", np.sum((std_before - std_after).cpu().detach().numpy()))

        return ret

    def recluster(self, Conv_input):
        N, C, W, H = Conv_input.size()
        t = time.time()
        self.indexes = self.SimilarityGroupNormClustering(clone(Conv_input), self.num_groups, self.layer_index).to(
            dtype=torch.int64)
        # print(f"SimilarityGroupNormClustering time {time.time() - t}")

        self.reverse_indexes = empty_like(self.indexes)
        for ind in range(N*C):
            self.reverse_indexes[self.indexes[ind]] = tensor(ind, device=global_vars.device)

    def SimilarityGroupNormClustering(self, channels_input, numGruops, layerIndex):
        N, C, H, W = channels_input.size()
        groupSize = int(C / numGruops)
        input_no_h_w = channels_input.reshape(N * C, H * W)

        if not global_vars.args.use_k_means:
            # N, C, H = input_no_h_w.size()
            channel_dist = input_no_h_w #cat([input_no_h_w[:, i, :] for i in range(C)], dim=1)
            mean = channel_dist.mean(dim=1)
            var = channel_dist.var(dim=1)
            sort_metric = (mean / var) * (mean + var)
            sorted_indexes = sorted(range(len(sort_metric)), key=lambda k: sort_metric[k])

            if global_vars.args.far_groups:
                endlist = [[] for _ in range(numGruops)]
                for index, item in enumerate(sorted_indexes):
                    endlist[index % numGruops].append(item)

                new_list = []
                for i in range(len(endlist)):
                    new_list = new_list + endlist[i]

                sorted_indexes = new_list
            channelsClustering = torch.Tensor(sorted_indexes)  # zeros_like(sort_metric, device=global_vars.device)


        else:
            clf = KMeansConstrained(
                n_clusters=int(numGruops),
                size_min=groupSize,
                size_max=groupSize,
                random_state=global_vars.args.seed
            )
            norm_df, df = self.get_df(input_no_h_w)

            np_df = norm_df.detach().cpu().numpy()
            s = time.time()
            np_clusters = clf.fit_predict(np_df)
            # print(f"fit_predict time = {time.time() - s}")
            clusters = tensor(np_clusters, device=global_vars.device)

            indexes = sort(clusters)[1]
            channelsClustering = zeros_like(clusters, device=global_vars.device)
            for g in range(numGruops):
                for i in range(groupSize):
                    channelsClustering[indexes[g * groupSize + i]] = g * groupSize + i

        #         torch.save(
        #           {
        #             'layer': self.layer_index,
        #             'epoch': global_vars.normalizationEpoch,
        #             'df': df,
        #             'channelsClustering': channelsClustering,
        #           },
        #           '/content/drive/MyDrive/AdaptiveNormalization/results/SGN channels data/'
        #               +'layer{}_epoch{}.tar'.format(self.layer_index, global_vars.normalizationEpoch)
        #         )
        return channelsClustering

    def get_df(self, channels_input):
        N, C, H = channels_input.size()
        channel_dist = cat([channels_input[i, :, :] for i in range(N)], dim=1)
        mean = channel_dist.mean(dim=1).reshape(C, 1)
        var = channel_dist.var(dim=1).reshape(C, 1)
        std = channel_dist.std(dim=1).reshape(C, 1)
        Vsize = H
        sorted_channel_dist, _ = sort(channel_dist, dim=1)
        med1 = int(ceil(Vsize * tensor([0.25])))
        med2 = int(floor(Vsize * tensor([0.5])))
        med3 = int(floor(Vsize * tensor([0.75])))
        madian3 = sorted_channel_dist[:, [med1, med2, med3]].to(device=global_vars.device)
        df = cat([mean, var, std, madian3], dim=1)  #
        norm_df = df.clone()
        for i in range(norm_df.shape[1]):
            norm_df[:, i] = normalize(norm_df[:, i], dim=-1, eps=self.eps)
        return norm_df, df
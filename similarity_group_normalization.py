import sys
import time

import numpy as np
import torch
from torch import clone, empty_like, tensor, sort, zeros_like, cat, ceil, floor
from torch.nn import Module, GroupNorm
import global_vars
from agn_utils import getLayerIndex
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

        if global_vars.args.epoch_start_cluster >= global_vars.epoch_num:
            return self.groupNorm(Conv_input)

        N, C, H, W = Conv_input.size()

        if global_vars.args.plot_std:
            input_2_dim = Conv_input.reshape(N * C, H * W)
            std_before = input_2_dim.std(dim=1)
            self.before_list.append(torch.sum(std_before).cpu().detach().numpy())

        if global_vars.recluster:
            self.recluster(Conv_input)
            self.batch_layer_index[batch_num] = (self.indexes.clone(), self.reverse_indexes.clone())
        Conv_input_reshaped = Conv_input.reshape(N * C, W * H)

        if global_vars.args.save_shuff_idxs:  # not used
            Conv_input_new_idx = Conv_input_reshaped[self.batch_layer_index[batch_num][0], :]
            Conv_input_new_idx_norm = self.groupNorm(Conv_input_new_idx.reshape(N, C, H, W))
            Conv_input_new_idx_norm = Conv_input_new_idx_norm.reshape(N*C, W*H)
            Conv_input_orig_idx_norm = Conv_input_new_idx_norm[self.batch_layer_index[batch_num][1], :]
        else:
            Conv_input_new_idx = Conv_input_reshaped[self.indexes, :]
            GN_input = Conv_input_new_idx.reshape(N, C, H, W)
            Conv_input_new_idx_norm = self.groupNorm(GN_input)
            Conv_input_new_idx_norm = Conv_input_new_idx_norm.reshape(N*C, W*H)
            Conv_input_orig_idx_norm = Conv_input_new_idx_norm[self.reverse_indexes, :]

        ret = Conv_input_orig_idx_norm.reshape(N, C, H, W).requires_grad_(requires_grad=True)

        if global_vars.args.plot_std:
            N, C, H, W = ret.size()
            input_2_dim = ret.reshape(N * C, H * W)
            std_after = input_2_dim.std(dim=1)
            self.before_list.append(torch.sum(std_after).cpu().detach().numpy())

            print("std before - after:", np.sum((std_before - std_after).cpu().detach().numpy()))

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
        # grouping as far channels, (mean/var)*(mean+var), range in group: numGruops
        if global_vars.args.SGN_version == 1:
            channelsClustering = self.SortChannelsV1(channels_input, numGruops)

        # grouping as far channels, (mean/var)*(mean+var), range in group: groupSize/numberOfChannels
        elif global_vars.args.SGN_version == 2:
            channelsClustering = self.SortChannelsV2(channels_input, groupSize)

        # grouping as close channels, (mean/var)*(mean+var)
        elif global_vars.args.SGN_version == 3:
            channelsClustering = self.SortChannelsV3(channels_input, groupSize)

        # grouping as close channels, KMeans
        elif global_vars.args.SGN_version == 4:
            channelsClustering = self.SortChannelsV4(channels_input, numGruops)

        # groups only in the same batch[i], (mean/var)*(mean+var)
        elif global_vars.args.SGN_version == 5:
            channelsClustering = self.SortChannelsV5(channels_input, numGruops)

        elif global_vars.args.SGN_version == 6:
            channelsClustering = self.SortChannelsV6(channels_input)

        # grouping using diffusion maps
        elif global_vars.args.SGN_version == 7:
            channelsClustering = self.SortChannelsV7(channels_input, numGruops)

        # grouping using harmonic mean
        elif global_vars.args.SGN_version == 8:
            channelsClustering = self.SortChannelsV8(channels_input)

        else:
            print(f"SGN_version number {global_vars.args.SGN_version} is not available!")
            exit(1)

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
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)
        channel_dist = input_2_dim # cat([channels_input[i, :, :] for i in range(N)], dim=1)
        mean = channel_dist.mean(dim=1)#.reshape(C, 1)
        var = channel_dist.var(dim=1)#.reshape(C, 1)
        std = channel_dist.std(dim=1)#.reshape(C, 1)
        Vsize = H*W
        sorted_channel_dist, _ = sort(channel_dist, dim=1)
        med1 = int(ceil(Vsize * tensor([0.25])))
        med2 = int(floor(Vsize * tensor([0.5])))
        med3 = int(floor(Vsize * tensor([0.75])))
        madian3 = sorted_channel_dist[:, [med1, med2, med3]].to(device=global_vars.device)
        df = cat([mean, var, std], dim=0)  #
        norm_df = df.clone()
        # for i in range(norm_df.shape[1]):
        #     norm_df[:, i] = normalize(norm_df[:, i], dim=-1, eps=self.eps)
        return norm_df, df

    # grouping as far channels, (mean/var)*(mean+var), range in group: numGruops
    def SortChannelsV1(self, channels_input, numGruops):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)

        channel_dist = input_2_dim  # cat([input_no_h_w[:, i, :] for i in range(C)], dim=1)
        mean = channel_dist.mean(dim=1)
        var = channel_dist.var(dim=1)
        sort_metric = (mean / var) * (mean + var)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])

        endlist = [[] for _ in range(numGruops)]
        for index, item in enumerate(sorted_indexes):
            endlist[index % numGruops].append(item)

        new_list = []
        for i in range(len(endlist)):
            new_list = new_list + endlist[i]

        sorted_indexes = new_list
        channelsClustering = torch.Tensor(sorted_indexes)  # zeros_like(
        # sort_metric, device=global_vars.device)

        return channelsClustering

    # grouping as far channels, (mean/var)*(mean+var), range in group:
    # numberOfChannels/groupSize
    def SortChannelsV2(self, channels_input, groupSize):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)
        channel_dist = input_2_dim  # cat([input_no_h_w[:, i, :] for i in range(
        # C)], dim=1)
        mean = channel_dist.mean(dim=1)
        var = channel_dist.var(dim=1)
        sort_metric = (mean / var) * (mean + var)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])
        range_in_group = N * C // groupSize
        endlist = [[] for _ in range(range_in_group)]
        for index, item in enumerate(sorted_indexes):
            endlist[index % range_in_group].append(item)

        new_list = []
        for i in range(len(endlist)):
            new_list = new_list + endlist[i]

        sorted_indexes = new_list
        channelsClustering = torch.Tensor(sorted_indexes)  # zeros_like(
        # sort_metric, device=global_vars.device)

        return channelsClustering

    # grouping as close channels, (mean/var)*(mean+var)
    def SortChannelsV3(self, channels_input, groupSize):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)
        mean = input_2_dim.mean(dim=1)
        var = input_2_dim.var(dim=1)
        sort_metric = (mean / var) * (mean + var)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])
        channelsClustering = torch.Tensor(sorted_indexes)

        return channelsClustering

    def SortChannelsV4(self, channels_input, numGruops):
        from k_means_constrained import KMeansConstrained
        N, C, H, W = channels_input.size()
        input_no_h_w = channels_input.reshape(N * C, H * W)
        groupSize = int(C / numGruops)
        clf = KMeansConstrained(
            n_clusters=int(numGruops),
            # size_min=groupSize,
            # size_max=groupSize,
            random_state=global_vars.args.seed
        )
        norm_df, df = self.get_df(channels_input)

        np_df = norm_df.detach().cpu().numpy()
        np_clusters = clf.fit_predict(np_df)
        clusters = tensor(np_clusters, device=global_vars.device)

        indexes = sort(clusters)[1]
        channelsClustering = zeros_like(clusters, device=global_vars.device)
        for g in range(numGruops):
            for i in range(groupSize):
                channelsClustering[indexes[g * groupSize + i]] = g * groupSize + i

        return channelsClustering

    # groups only in the same batch[i], (mean/var)*(mean+var)
    def SortChannelsV5(self, channels_input, groupSize):
        N, C, H, W = channels_input.size()
        input_N_C_WH = channels_input.reshape(N, C, H * W)
        mean = input_N_C_WH.mean(dim=2)
        var = input_N_C_WH.var(dim=2)
        sort_metric = (mean / var) * (mean + var)
        channelsClustering = torch.Tensor()
        for b in range(N):
            channelsClustering = cat((channelsClustering, torch.Tensor(
                sorted(range(len(sort_metric[b, :])),
                       key=lambda k: sort_metric[b][k])) + (C * b)))
        return channelsClustering

    # grouping as close channels, (std)
    def SortChannelsV6(self, channels_input):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)
        sort_metric = input_2_dim.std(dim=1)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])
        channelsClustering = torch.Tensor(sorted_indexes)

        return channelsClustering

    def SortChannelsV7(self, channels_input, numGruops):
        from sklearn.metrics.pairwise import pairwise_distances
        from sklearn.manifold import SpectralEmbedding

        # Reshape the tensor to be 2D (num_channels x num_pixels)
        num_channels = channels_input.shape[0] * channels_input.shape[1]
        num_pixels = channels_input.shape[2] * channels_input.shape[3]
        data = channels_input.reshape(num_channels, num_pixels)

        # Compute pairwise distance matrix
        D = pairwise_distances(data.detach().numpy(), metric='euclidean')

        # Compute affinity matrix
        gamma = 1.0 / (2 * np.median(D) ** 2)
        W = np.exp(-gamma * D ** 2)

        # Apply SpectralEmbedding
        n_components = 1
        embedder = SpectralEmbedding(n_components=n_components,
                                     affinity='precomputed')
        X_embedded = embedder.fit_transform(W)

        x_embedded = X_embedded.reshape(-1)

        sorted_indexes = np.argsort(x_embedded)

        return torch.from_numpy(sorted_indexes)

    def SortChannelsV8(self, channels_input):
        N, C, H, W = channels_input.size()
        input_N_C_WH = channels_input.reshape(N * C, H * W)
        mean = input_N_C_WH.mean(dim=1)
        var = input_N_C_WH.var(dim=1)

        # harmonic mean
        sort_metric = 2 * (mean * var) / (mean + var)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])
        channelsClustering = torch.Tensor(sorted_indexes)

        return channelsClustering


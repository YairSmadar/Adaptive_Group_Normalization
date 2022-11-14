import time

from torch import clone, empty_like, tensor, sort, zeros_like, cat, ceil, floor
from torch.nn import Module, GroupNorm
from torch.nn.functional import normalize
import torch
import global_vars
from agn_utils import getLayerIndex
from random_group_normalization import better_places
from k_means_constrained import KMeansConstrained


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

    def forward(self, Conv_input):
        if global_vars.recluster:
            self.recluster(Conv_input)
            # raise Exception("stop")

        ret = self.groupNorm(Conv_input[:, self.indexes, :, :])[:, self.reverse_indexes, :, :].requires_grad_(
            requires_grad=True)

        return ret

    def recluster(self, Conv_input):
        # s = time.time()
        self.indexes = self.SimilarityGroupNormClustering(clone(Conv_input), self.num_groups, self.layer_index).to(
            dtype=torch.int64)
        # print(f"SimilarityGroupNormClustering time = {time.time() - s}")
        #         self.indexes = better_places(self.indexes, self.num_groups, self.group_size, self.num_channels)
        self.reverse_indexes = empty_like(self.indexes)
        for ind in range(Conv_input.size()[1]):
            self.reverse_indexes[self.indexes[ind]] = tensor(ind, device=global_vars.device)

    def SimilarityGroupNormClustering(self, channels_input, numGruops, layerIndex):
        N, C, H, W = channels_input.size()
        groupSize = int(C / numGruops)
        input_no_h_w = channels_input.reshape(N, C, H * W)

        if True:
            N, C, H = input_no_h_w.size()
            channel_dist = cat([input_no_h_w[i, :, :] for i in range(N)], dim=1)
            mean = channel_dist.mean(dim=1)
            # var = channel_dist.var(dim=1)
            mean_plus_var = mean  # + var
            sorted_indexes = sorted(range(len(mean_plus_var)), key=lambda k: mean_plus_var[k])
            channelsClustering = zeros_like(mean_plus_var, device=global_vars.device)
            for g in range(numGruops):
                for i in range(groupSize):
                    channelsClustering[sorted_indexes[g * groupSize + i]] = g * groupSize + i
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
            print(f"fit_predict time = {time.time() - s}")
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

import math

import numpy as np
import torch
from torch import clone, tensor, sort, zeros_like, cat, ceil, floor
from torch.nn import Module, GroupNorm
import global_vars
from agn_utils import getLayerIndex
import heapq
from scipy.cluster.hierarchy import linkage, leaves_list
from k_means_constrained import KMeansConstrained
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from abc import ABC, abstractmethod

if global_vars.args.plot_groups:
    import matplotlib.pyplot as plt


class SimilarityGroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int = 32, eps=1e-12,
                 strategy=None, no_shuff_best_k_p: float = 1.0):
        super(SimilarityGroupNorm, self).__init__()
        self.groupNorm = GroupNorm(num_groups, num_channels, eps=eps,
                                   affine=True)
        self.indexes = None
        self.reverse_indexes = None
        self.groups_representation_num = None  # not used
        self.eval_indexes = None
        self.eval_reverse_indexes = None
        self.layer_index = getLayerIndex()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_size = int(num_channels / num_groups)
        if no_shuff_best_k_p != 1.0 and \
                math.floor(self.num_groups * no_shuff_best_k_p) != 0:
            self.keep_best_std_groups = True
            self.filtered_num_groups = \
                num_groups - math.floor(self.num_groups * no_shuff_best_k_p)
        else:
            self.keep_best_std_groups = False
        self.eps = eps
        self.strategy = strategy
        self.no_shuff_best_k_p = no_shuff_best_k_p

    def forward(self, Conv_input):

        # start shuffle at epoch > 0
        if global_vars.args.epoch_start_cluster > global_vars.epoch_num:
            return self.groupNorm(Conv_input)

        N, C, H, W = Conv_input.size()

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
        self.indexes = self.SimilarityGroupNormClustering(clone(Conv_input)).to(
            dtype=torch.int64)

        self.reverse_indexes = torch.argsort(self.indexes).to(
            Conv_input.device)

    def SimilarityGroupNormClustering(self, channels_input):

        N, C, _, _ = channels_input.size()
        if self.strategy is not None:
            if self.keep_best_std_groups:
                best_std_groups = self.find_best_std_groups(channels_input)
                filtered_channels_input, included_channels = self.exclude_std_groups(channels_input, best_std_groups)

                self.strategy.filtered_num_groups = self.filtered_num_groups

                best_group_size = best_std_groups.size()[0]
                N_best_groups = torch.empty((best_group_size * N), dtype=torch.long).to(channels_input.device)

                for i in range(N):
                    s = best_group_size*i
                    e = s + best_group_size
                    N_best_groups[s:e] = best_std_groups + (self.num_groups * i)

            else:
                filtered_channels_input = channels_input

            channelsClustering = self.strategy.sort_channels(filtered_channels_input)

        else:
            print("No clustering strategy defined!")
            exit(1)

        self.get_channels_clustering_for_eval(filtered_channels_input, channelsClustering)

        if self.keep_best_std_groups:

            if self.indexes is None:
                self.indexes = torch.arange(0, self.num_groups * self.group_size * N).to(channels_input.device)

            # Creating a mask to select the channels to be re-clustered
            mask = torch.ones(self.num_groups*N, dtype=torch.bool).to(channels_input.device)

            for idx in N_best_groups:
                mask[idx] = 0

            # Reshaping mask to align with channel indices and repeating it according to group size
            mask = mask.unsqueeze(-1).repeat(1, self.group_size).view(-1)

            # Get the channels to be re-clustered
            channels_to_cluster = self.indexes[mask]

            # Creating a tensor for the new order of channels
            new_indexes = torch.empty_like(self.indexes).to(channels_input.device)

            # Instead of shifting all channels, now we determine the shift individually for each channel
            # Depending on how many unchanged groups were in front of its original group
            original_group_of_channel = torch.div(channels_to_cluster, self.group_size, rounding_mode='floor')

            num_unclustered_before = (original_group_of_channel.unsqueeze(-1) > N_best_groups.unsqueeze(0)).sum(dim=1)
            shifts = (num_unclustered_before * self.group_size).to(channels_input.device)
            new_indexes[mask] = channelsClustering + shifts

            # Placing the non-reclustered groups back in their original positions with original values
            new_indexes[~mask] = self.indexes[~mask]

            channelsClustering = new_indexes

        return channelsClustering

    def exclude_std_groups(self, channels_input, best_std_groups):
        excluded_channels = []
        for group_id in range(self.num_groups):
            if group_id not in best_std_groups:
                group_start = group_id * self.group_size
                group_end = group_start + self.group_size
                excluded_channels.extend(list(range(group_start, group_end)))

        excluded_channels = torch.tensor(excluded_channels).long().to(
            channels_input.device)

        channels_input_excluded = channels_input[:, excluded_channels, :, :]

        return channels_input_excluded, excluded_channels

    def find_best_std_groups(self, channels_input):
        with torch.no_grad():
            N, C, H, W = channels_input.size()
            channels_input_groups = channels_input.view(N, self.num_groups,
                                                        torch.div(C,
                                                                  self.num_groups,
                                                                  rounding_mode='floor'),
                                                        H, W)

            stds = torch.std(channels_input_groups, dim=(0, 2, 3, 4))
            no_shuff_best_k = math.floor(self.num_groups * self.no_shuff_best_k_p)
            values, indices = torch.topk(stds, no_shuff_best_k)
            return indices.sort().values

    def get_channels_clustering_for_eval(self, channels_input: torch.Tensor,
                                         channelsClustering: torch.Tensor):

        # in case the channels are shuffle the same for every image,
        # We used this from version 10 and above
        if global_vars.args.SGN_version >= 10:
            self.eval_indexes = channelsClustering
            self.eval_reverse_indexes = torch.argsort(self.eval_indexes).to(
                channels_input.device)

            return

        self.eval_indexes = \
            self.strategy.select_channels_indices_according_to_the_most(
                channels_input, channelsClustering)

        self.eval_reverse_indexes = torch.argsort(self.eval_indexes).to(
            channels_input.device)


class ClusteringStrategy(ABC):
    def __init__(self, num_groups: int, num_channels: int = 32):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_size = int(num_channels / num_groups)
        self.filtered_num_groups = num_groups

    @abstractmethod
    def sort_channels(self, channels_input):
        pass

    def plot_groups(self, channels_groups, means, vars):
        groups = torch.repeat_interleave(torch.arange(self.filtered_num_groups),
                                         torch.div(len(channels_groups),
                                                   self.filtered_num_groups,
                                                   rounding_mode='floor'),
                                         dim=0)

        # Create a scatter plot with points colored by group
        plt.figure(figsize=(10, 10))
        plt.scatter(means[channels_groups].detach().numpy(),
                    vars[channels_groups].detach().numpy(), c=groups,
                    cmap='tab20', alpha=0.5)

        # Optionally, add a colorbar
        plt.colorbar(label='Group')

        plt.show()

    def harmonic_mean(self, _tensor, dim=1):
        """
        Calculate the harmonic mean of 2 tensors, for the var and mean of
        the tensor.
        """
        if _tensor.dim() == 4:
            N, C, H, W = _tensor.size()

            if dim == 1:
                target_tensor = _tensor.reshape(N * C, H * W)
            else:
                target_tensor = _tensor

            mean = target_tensor.mean(dim=dim)
            var = target_tensor.var(dim=dim)

        elif _tensor.dim() == 2:
            mean = _tensor.mean()
            var = _tensor.var()
        else:
            raise Exception('harmonic mean support 4 or 2 dim only')

        return 2 * (mean * var) / (mean + var)

    def select_channels_indices_according_to_the_most(
            self,
            channels_input: torch.Tensor,
            channelsClustering: torch.Tensor):

        N, C, _, _ = channels_input.size()

        channel_groups = {i: [] for i in range(C)}
        for i in range(N * C):
            channel_to = channelsClustering[i] % C
            group_num = self.map_to_group(i)
            channel_groups[channel_to.item()].append(group_num)

        max_elements = self.get_num_occurrences(channel_groups)

        final_channel_groups = {i: [] for i in range(self.filtered_num_groups)}

        # Create a max heap (using negative values)
        min_heap = [(-max_val, channel_num, group) for channel_num, max_vals in
                    enumerate(max_elements.values()) for group, max_val in
                    enumerate(max_vals)]

        # Heapify the max heap
        heapq.heapify(min_heap)
        added_indices = set()
        while True:
            # Get the corresponding index and group of the maximum value
            neg_max_value, channel_num, group = heapq.heappop(min_heap)

            if len(final_channel_groups[
                       group]) < self.group_size and channel_num not in added_indices:
                final_channel_groups[group].append(channel_num)
                added_indices.add(channel_num)

            # Break the loop when all groups are filled
            if all(len(lst) == self.group_size for lst in
                   final_channel_groups.values()):
                break

        # Flatten the lists in the dictionary
        flat_list = [elem for lst in final_channel_groups.values() for elem in
                     lst]

        # Convert the flattened list to a PyTorch tensor
        factors = torch.arange(0, N) * C

        eval_indexes = torch.tensor(flat_list)
        eval_indexes = \
            eval_indexes.repeat(N).reshape(N, C) + factors.unsqueeze(1)
        eval_indexes = eval_indexes.reshape(-1).to(channels_input.device)

        return eval_indexes

    def map_to_group(self, X: int):
        group_num = torch.div(X,
                              self.group_size,
                              rounding_mode='floor') % self.filtered_num_groups
        return group_num

    def get_num_occurrences(self, d):
        num_counts = {i: [0] * self.filtered_num_groups for i in range(len(d))}

        for i, lst in enumerate(d.values()):
            for num in lst:
                num_counts[i][num] += 1

        return num_counts

    def create_shuff_for_total_batch(self, channels_input,
                                     channelsClustering):
        N, C, H, W = channels_input.size()
        factors = (torch.arange(0, N) * C).to(channels_input.device)
        channelsClustering = torch.cat([channelsClustering] * N)
        channelsClustering = channelsClustering.to(channels_input.device)
        channelsClustering = \
            (channelsClustering.reshape(N, C) + factors.unsqueeze(1)).view(
                -1)

        return channelsClustering

    def KMeans_2D(self, channel_vars, channel_means):

        # Create a 2D tensor where each row is a channel
        # and the columns are the mean and variance
        channel_stats = torch.stack((channel_means, channel_vars), dim=1)

        # Perform constrained k-means clustering on the channel statistics
        kmeans = KMeansConstrained(n_clusters=self.filtered_num_groups,
                                   size_min=self.group_size,
                                   size_max=self.group_size,
                                   random_state=global_vars.args.seed)
        kmeans.fit(channel_stats.cpu().detach().numpy())

        # The labels_ attribute of the fitted model
        # gives the group for each channel
        groups = kmeans.labels_

        # Get the indices that would sort the groups
        new_order = np.argsort(groups)

        return new_order

    def SortChannelsOutliersKMeans(self, channels_input, method='IsolationForest'):
        # Convert to a numpy array
        feature_vecs_np = self.create_mean_var_nparray(channels_input)

        # Indices of inliers/outliers
        if method == 'IsolationForest':
            outliers = self.get_outliers_IsolationForest(feature_vecs_np)
        elif method == 'ZScoreV1':
            # Calculate the centroid
            centroid = np.mean(feature_vecs_np, axis=0)
            # Calculate the Euclidean distance from each point to the centroid
            distances = np.linalg.norm(feature_vecs_np - centroid, axis=1)
            # Calculate the z-score of the distances
            z_scores = zscore(distances)
            # Absolute Z-scores > 3 are considered as outliers
            outliers = np.abs(z_scores) > 3

            outliers = np.where(outliers)[0]
        elif method == 'ZScoreV2':
            # Calculate Z-scores
            z_scores = zscore(feature_vecs_np)

            # Absolute Z-scores > 3 are considered as outliers
            outliers = np.abs(z_scores) > 3

            outliers = np.where(outliers)[0]
        else:
            raise ValueError(
                "Invalid method. Expected 'IsolationForest' or 'ZScore'")

        mean_vals, var_vals = self.get_mean_val_no_outliers(outliers,
                                                            channels_input)

        new_order = self.KMeans_2D(var_vals, mean_vals)

        ret = self.create_shuff_for_total_batch(channels_input,
                                                torch.from_numpy(new_order))

        if global_vars.args.plot_groups:
            channel_means = channels_input.mean(dim=(0, 2, 3))
            channel_vars = channels_input.var(dim=(0, 2, 3))

            self.plot_groups(new_order, channel_means, channel_vars)

        return ret.to(channels_input.device)

    def get_mean_val_no_outliers(self, outliers, channels_input):

        _, C, _, _ = channels_input.size()

        # calculate the N and C indices for each element of the outliers tensor
        N_indices = torch.div(outliers,
                              C,
                              rounding_mode='floor')
        C_indices = outliers % C

        # create a mask of ones with the same shape as the input
        mask = torch.ones_like(channels_input).to(channels_input.device)

        # convert N_indices and C_indices to tensors
        N_indices = torch.tensor(N_indices).to(channels_input.device)
        C_indices = torch.tensor(C_indices).to(channels_input.device)

        for i in range(len(outliers)):
            mask[N_indices[i], C_indices[i], :, :] = 0

        # calculate the number of valid (non-zero) values for each channel
        valid_counts = torch.sum(mask, dim=(0, 2, 3))
        valid_counts[valid_counts == 0] = 1

        # compute the sum of the non-outlier channels
        sum_vals = torch.sum(channels_input * mask, dim=(0, 2, 3))

        # compute the mean of the non-outlier channels
        mean_vals = sum_vals / valid_counts

        # compute the variance of the non-outlier channels
        var_vals = torch.sum(
            mask * (channels_input - mean_vals.view(1, -1, 1, 1)) ** 2,
            dim=(0, 2, 3)) / valid_counts

        return mean_vals, var_vals

    def create_mean_var_nparray(self, channels_input):
        mean_vals = torch.mean(channels_input, dim=(2, 3))
        var_vals = torch.var(channels_input, dim=(2, 3))

        # Reshape to (N*C)
        mean_vals = mean_vals.view(-1,1)
        var_vals = var_vals.view(-1,1)

        # Concatenate mean_vals and var_vals along the channel dimension
        feature_vecs = torch.cat((mean_vals, var_vals), dim=-1)

        # Convert to a numpy array
        feature_vecs_np = feature_vecs.cpu().detach().numpy()

        return feature_vecs_np

    def get_outliers_IsolationForest(self, feature_vecs_np):
        # Assuming feature_vecs_np is your data (mean, var pairs)
        iso_forest = IsolationForest(
            contamination=0.1)  # adjust contamination as needed
        outliers = iso_forest.fit_predict(feature_vecs_np)

        # Indices of inliers
        outliers = np.where(outliers == -1)[0]

        return outliers


class SortChannelsV1(ClusteringStrategy):
    def sort_channels(self, channels_input):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)

        channel_dist = input_2_dim  # cat([input_no_h_w[:, i, :] for i in range(C)], dim=1)
        mean = channel_dist.mean(dim=1)
        var = channel_dist.var(dim=1)
        sort_metric = (mean / var) * (mean + var)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])

        endlist = [[] for _ in range(self.filtered_num_groups)]
        for index, item in enumerate(sorted_indexes):
            endlist[index % self.filtered_num_groups].append(item)

        new_list = []
        for i in range(len(endlist)):
            new_list = new_list + endlist[i]

        sorted_indexes = new_list
        channelsClustering = torch.Tensor(sorted_indexes)  # zeros_like(
        # sort_metric, device=global_vars.device)

        return channelsClustering


class SortChannelsV2(ClusteringStrategy):
    def sort_channels(self, channels_input):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)
        channel_dist = input_2_dim  # cat([input_no_h_w[:, i, :] for i in range(
        # C)], dim=1)
        mean = channel_dist.mean(dim=1)
        var = channel_dist.var(dim=1)
        sort_metric = (mean / var) * (mean + var)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])
        range_in_group = torch.div(N*C,
                                   self.group_size, 
                                   rounding_mode='floor')
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


class SortChannelsV3(ClusteringStrategy):
    def sort_channels(self, channels_input):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)
        mean = input_2_dim.mean(dim=1)
        var = input_2_dim.var(dim=1)
        sort_metric = (mean / var) * (mean + var)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])
        channelsClustering = torch.Tensor(sorted_indexes)

        return channelsClustering


class SortChannelsV4(ClusteringStrategy):
    def sort_channels(self, channels_input):
        from k_means_constrained import KMeansConstrained
        N, C, H, W = channels_input.size()
        input_no_h_w = channels_input.reshape(N * C, H * W)
        clf = KMeansConstrained(
            n_clusters=int(self.filtered_num_groups),
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
        for g in range(self.filtered_num_groups):
            for i in range(self.group_size):
                channelsClustering[
                    indexes[g * self.group_size + i]] = g * self.group_size + i

        return channelsClustering

    def get_df(self, channels_input):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)
        channel_dist = input_2_dim  # cat([channels_input[i, :, :] for i in range(N)], dim=1)
        mean = channel_dist.mean(dim=1)  # .reshape(C, 1)
        var = channel_dist.var(dim=1)  # .reshape(C, 1)
        std = channel_dist.std(dim=1)  # .reshape(C, 1)
        Vsize = H * W
        sorted_channel_dist, _ = sort(channel_dist, dim=1)
        med1 = int(ceil(Vsize * tensor([0.25])))
        med2 = int(floor(Vsize * tensor([0.5])))
        med3 = int(floor(Vsize * tensor([0.75])))
        madian3 = sorted_channel_dist[:, [med1, med2, med3]].to(
            device=global_vars.device)
        df = cat([mean, var, std], dim=0)  #
        norm_df = df.clone()
        # for i in range(norm_df.shape[1]):
        #     norm_df[:, i] = normalize(norm_df[:, i], dim=-1, eps=self.eps)
        return norm_df, df


class SortChannelsV5(ClusteringStrategy):
    def sort_channels(self, channels_input):
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


class SortChannelsV6(ClusteringStrategy):
    def sort_channels(self, channels_input):
        N, C, H, W = channels_input.size()
        input_2_dim = channels_input.reshape(N * C, H * W)
        sort_metric = input_2_dim.std(dim=1)
        sorted_indexes = sorted(range(len(sort_metric)),
                                key=lambda k: sort_metric[k])
        channelsClustering = torch.Tensor(sorted_indexes)

        return channelsClustering


class SortChannelsV7(ClusteringStrategy):
    def sort_channels(self, channels_input):
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


class SortChannelsV8(ClusteringStrategy):
    def sort_channels(self, channels_input):
        sort_metric = self.harmonic_mean(channels_input)

        channelsClustering = torch.argsort(sort_metric)
        if global_vars.args.plot_groups:
            N, C, H, W = channels_input.size()
            t = channels_input.reshape(N * C, H * W)
            channel_means = t.mean(dim=1)
            channel_vars = t.var(dim=1)

            self.plot_groups(channelsClustering, channel_means, channel_vars)

        return channelsClustering


class SortChannelsV9(ClusteringStrategy):
    def sort_channels(self, channels_input):
        N, C, H, W = channels_input.size()
        channelsClustering = torch.zeros((N * C))

        for b in range(N):
            sort_metric = self.harmonic_mean(channels_input[b].view(1, C, H, W))
            channelsClustering[b * C:(b + 1) * C] = torch.argsort(sort_metric)

        factors = torch.arange(0, N) * C
        channelsClustering = \
            (channelsClustering.reshape(N, C) + factors.unsqueeze(1)).view(-1)

        return channelsClustering.to(channels_input.device)


class SortChannelsV10(ClusteringStrategy):
    def sort_channels(self, channels_input):
        N, C, H, W = channels_input.size()

        sort_metric = self.harmonic_mean(_tensor=channels_input, dim=(0, 2, 3))
        order = torch.argsort(sort_metric)

        factors = (torch.arange(0, N) * C).to(channels_input.device)
        channelsClustering = torch.cat([order] * N)
        channelsClustering = channelsClustering.to(channels_input.device)
        channelsClustering = \
            (channelsClustering.reshape(N, C) + factors.unsqueeze(1)).view(-1)

        if global_vars.args.plot_groups:
            channel_means = channels_input.mean(dim=(0, 2, 3))
            channel_vars = channels_input.var(dim=(0, 2, 3))

            self.plot_groups(order, channel_means, channel_vars)

        return channelsClustering.to(channels_input.device)


class SortChannelsV11(ClusteringStrategy):
    def sort_channels(self, channels_input):
        _SortChannelsV10 = SortChannelsV10(self.filtered_num_groups, self.num_channels)
        channelsClustering = _SortChannelsV10.sort_channels(channels_input)
        channelsClustering = self.select_channels_indices_according_to_the_most(
            channels_input, channelsClustering
        )

        return channelsClustering


class SortChannelsV12(ClusteringStrategy):
    def sort_channels(self, channels_input):
        N, C, W, H = channels_input.size()
        # Calculate the mean and variance for each channel
        channel_means = torch.mean(channels_input, dim=(0, 2, 3))
        channel_vars = torch.var(channels_input, dim=(0, 2, 3))

        # Stack the means and vars together to form a new tensor of shape (C, 2)
        channels = torch.stack((channel_means, channel_vars), dim=1)

        # Calculate the pairwise Euclidean distance
        distances = torch.cdist(channels, channels, p=2)

        # Fill the diagonal with a large value
        distances.fill_diagonal_(float('inf'))

        # Convert the distance matrix to a condensed distance matrix
        # (i.e., a flat array containing the upper triangular of the distance matrix)
        distances_condensed = distances[np.triu_indices(C, k=1)].cpu().detach().numpy()

        # Perform hierarchical/agglomerative clustering
        linkage_matrix = linkage(distances_condensed, method='average')

        # Get the order of channels
        order = leaves_list(linkage_matrix)

        ret = self.create_shuff_for_total_batch(channels_input, torch.from_numpy(order))

        if global_vars.args.plot_groups:
            self.plot_groups(order, channel_means, channel_vars)

        return ret.to(channels_input.device)


class SortChannelsV13(ClusteringStrategy):
    def sort_channels(self, channels_input):
        # Calculate the mean and variance for each channel
        channel_means = torch.mean(channels_input, dim=(0, 2, 3))
        channel_vars = torch.var(channels_input, dim=(0, 2, 3))

        # Get the indices that would sort the groups
        new_order = self.KMeans_2D(channel_vars, channel_means)

        ret = self.create_shuff_for_total_batch(channels_input,
                                                torch.from_numpy(new_order))

        if global_vars.args.plot_groups:
            channel_means = channels_input.mean(dim=(0, 2, 3))
            channel_vars = channels_input.var(dim=(0, 2, 3))

            self.plot_groups(new_order, channel_means, channel_vars)

        return ret.to(channels_input.device)


class SortChannelsV14(ClusteringStrategy):
    def sort_channels(self, channels_input):
        return self.SortChannelsOutliersKMeans(channels_input, method='IsolationForest')


class SortChannelsV15(ClusteringStrategy):
    def sort_channels(self, channels_input):
        return self.SortChannelsOutliersKMeans(channels_input, method='ZScoreV1')


class SortChannelsV16(ClusteringStrategy):
    def sort_channels(self, channels_input):
        return self.SortChannelsOutliersKMeans(channels_input, method='ZScoreV2')
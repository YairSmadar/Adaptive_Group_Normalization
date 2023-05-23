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

if global_vars.args.plot_groups:
    import matplotlib.pyplot as plt


class SimilarityGroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int = 32, eps=1e-12):
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
        self.eps = eps

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
        self.indexes = self.SimilarityGroupNormClustering(clone(Conv_input),
                                                          self.num_groups).to(
            dtype=torch.int64)

        self.reverse_indexes = torch.argsort(self.eval_indexes).to(
            Conv_input.device)

    def SimilarityGroupNormClustering(self, channels_input, numGruops):
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

        # grouping using harmonic mean, limit for shuff in the same image
        elif global_vars.args.SGN_version == 9:
            channelsClustering = self.SortChannelsV9(channels_input)

        # grouping using harmonic mean, limit for shuff the same in all images
        elif global_vars.args.SGN_version == 10:
            channelsClustering = self.SortChannelsV10(channels_input)

        # grouping using harmonic mean for each channel,
        # select most for each group, limit for shuff the same in all images
        elif global_vars.args.SGN_version == 11:
            channelsClustering = self.SortChannelsV11(channels_input)

        elif global_vars.args.SGN_version == 12:
            channelsClustering = self.SortChannelsV12(channels_input)

        elif global_vars.args.SGN_version == 13:
            channelsClustering = self.SortChannelsV13(channels_input)

        elif global_vars.args.SGN_version == 14:
            channelsClustering = self.SortChannelsV14(channels_input)


        else:
            print(
                f"SGN_version number {global_vars.args.SGN_version} is not available!")
            exit(1)

        self.get_channels_clustering_for_eval(channels_input,
                                              channelsClustering)

        return channelsClustering

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

        final_channel_groups = {i: [] for i in range(self.num_groups)}

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

    def get_channels_clustering_for_eval(self, channels_input: torch.Tensor,
                                         channelsClustering: torch.Tensor):

        # in case the channels are shuffle the same for every image
        if global_vars.args.SGN_version >= 10:
            self.eval_indexes = channelsClustering
            self.eval_reverse_indexes = torch.argsort(self.eval_indexes).to(
                channels_input.device)

            return

        self.eval_indexes = self.select_channels_indices_according_to_the_most(
            channels_input, channelsClustering
        )

        self.eval_reverse_indexes = torch.argsort(self.eval_indexes).to(
            channels_input.device)

    def remove_elements_with_same_value_from_heap(self, heap, i):
        elements_to_push_back = []

        while heap:
            neg_value, element_i, group = heapq.heappop(heap)
            if element_i != i:
                elements_to_push_back.append((neg_value, element_i, group))

        for element in elements_to_push_back:
            heapq.heappush(heap, element)

    def get_num_occurrences(self, d):
        num_counts = {i: [0] * self.num_groups for i in range(len(d))}

        for i, lst in enumerate(d.values()):
            for num in lst:
                num_counts[i][num] += 1

        return num_counts

    def map_to_group(self, X: int):
        group_num = (X // self.group_size) % self.num_groups
        return group_num

    def get_groups_representation_num(self, channels_input: torch.Tensor,
                                      channelsClustering: torch.Tensor):
        """
        Returns representative number for each group.
        Using for inference/Test
        """
        with torch.no_grad():
            N, C, H, W = channels_input.size()
            Conv_input_reshaped = channels_input.view(-1, W * H)

            # Use torch.index_select for better performance
            Conv_input_new_idx = torch.index_select(Conv_input_reshaped,
                                                    0, channelsClustering)
            GN_input = Conv_input_new_idx.view(N, C, H, W)

            # Reshape the tensor to (N, num_groups, group_size, H, W)
            GN_input_grouped = GN_input.view(N, self.num_groups,
                                             self.group_size, H, W)

            # Compute the mean and variance along the grouped channel dimension
            mean = GN_input_grouped.mean(dim=(2, 3, 4))
            var = GN_input_grouped.var(dim=(2, 3, 4))

            # Compute the harmonic mean
            sort_metric = 2 * (mean * var) / (mean + var)

            # Sum the sort_metric values across the batch and divide by N to get the mean
            self.groups_representation_num = \
                (sort_metric.sum(dim=0) / N).tolist()

    def eval_group_channels(self, tensor, metric):
        assert len(
            self.groups_representation_num) > 0, "The length of the representative_numbers list must be greater than 0."

        assert self.num_channels % self.num_groups == 0, "The number of channels must be divisible by the number of representative groups."

        with torch.no_grad():
            reverse_indices = []
            batch_size = tensor.shape[0]

            for b in range(batch_size):
                # Calculate the representative number for each channel
                channel_reps = torch.empty(self.num_channels,
                                           device=tensor.device)
                for i in range(self.num_channels):
                    channel_reps[i] = metric(tensor[b, i, :, :]).item()

                # Compute the distance between each channel and all representative numbers
                distances = torch.abs(channel_reps.view(-1, 1) - torch.tensor(
                    self.groups_representation_num, device=tensor.device))

                # Initialize the groups and available channels
                channel_groups = {i: [] for i in range(self.num_groups)}
                available_channels = torch.tensor(range(self.num_channels),
                                                  device=tensor.device)

                # Iteratively select the closest channel for each representative number
                for _ in range(self.num_channels // self.num_groups):
                    for group_idx in range(self.num_groups):
                        closest_channel_idx = torch.argmin(
                            distances[available_channels, group_idx])
                        channel_idx = available_channels[closest_channel_idx]
                        channel_groups[group_idx].append(channel_idx)
                        available_channels = torch.cat((available_channels[
                                                        :closest_channel_idx],
                                                        available_channels[
                                                        closest_channel_idx + 1:]))

                # Rearrange the tensor using the calculated indices
                batch_indices = torch.cat(
                    [torch.tensor(group, device=tensor.device) for group in
                     channel_groups.values()], dim=0)

                # Calculate the reverse indices
                batch_reverse_indices = torch.argsort(batch_indices)

                reverse_indices.append(batch_reverse_indices)

            reverse_indices_tensor = torch.stack(reverse_indices).to(
                tensor.device)

            return tensor, reverse_indices_tensor

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

    def create_shuff_for_total_batch(self, channels_input, channelsClustering):
        N, C, H, W = channels_input.size()
        factors = (torch.arange(0, N) * C).to(channels_input.device)
        channelsClustering = torch.cat([channelsClustering] * N)
        channelsClustering = channelsClustering.to(channels_input.device)
        channelsClustering = \
            (channelsClustering.reshape(N, C) + factors.unsqueeze(1)).view(-1)

        return channelsClustering

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
                channelsClustering[
                    indexes[g * groupSize + i]] = g * groupSize + i

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

        sort_metric = self.harmonic_mean(channels_input)

        channelsClustering = torch.argsort(sort_metric)
        if global_vars.args.plot_groups:
            N, C, H, W = channels_input.size()
            t = channels_input.reshape(N * C, H * W)
            channel_means = t.mean(dim=1)
            channel_vars = t.var(dim=1)

            self.plot_groups(channelsClustering, channel_means, channel_vars)

        return channelsClustering

    def SortChannelsV9(self, channels_input):

        N, C, H, W = channels_input.size()
        channelsClustering = torch.zeros((N * C))

        for b in range(N):
            sort_metric = self.harmonic_mean(channels_input[b].view(1, C, H, W))
            channelsClustering[b * C:(b + 1) * C] = torch.argsort(sort_metric)

        factors = torch.arange(0, N) * C
        channelsClustering = \
            (channelsClustering.reshape(N, C) + factors.unsqueeze(1)).view(-1)

        return channelsClustering.to(channels_input.device)

    def SortChannelsV10(self, channels_input):
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

    def SortChannelsV11(self, channels_input):
        channelsClustering = self.SortChannelsV8(channels_input)
        channelsClustering = self.select_channels_indices_according_to_the_most(
            channels_input, channelsClustering
        )

        return channelsClustering

    def SortChannelsV12(self, channels_input):

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

    def SortChannelsV13(self, channels_input):
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

    def SortChannelsV14(self, channels_input):

        N, C, W, H = channels_input.size()

        mean_vals = torch.mean(channels_input, dim=(2, 3))
        var_vals = torch.var(channels_input, dim=(2, 3))

        # Reshape to (N*C)
        mean_vals = mean_vals.view(-1,1)
        var_vals = var_vals.view(-1,1)

        # Concatenate mean_vals and var_vals along the channel dimension
        feature_vecs = torch.cat((mean_vals, var_vals), dim=-1)

        # Convert to a numpy array
        feature_vecs_np = feature_vecs.cpu().detach().numpy()

        # Assuming feature_vecs_np is your data (mean, var pairs)
        iso_forest = IsolationForest(
            contamination=0.1)  # adjust contamination as needed
        outliers = iso_forest.fit_predict(feature_vecs_np)

        # Indices of inliers
        outliers = np.where(outliers == -1)[0]

        # calculate the N and C indices for each element of the outliers tensor
        N_indices = outliers // C
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

        new_order = self.KMeans_2D(var_vals, mean_vals)

        ret = self.create_shuff_for_total_batch(channels_input,
                                                torch.from_numpy(new_order))

        if global_vars.args.plot_groups:
            channel_means = channels_input.mean(dim=(0, 2, 3))
            channel_vars = channels_input.var(dim=(0, 2, 3))

            self.plot_groups(new_order, channel_means, channel_vars)

        return ret.to(channels_input.device)

    def KMeans_2D(self, channel_vars, channel_means):

        # Create a 2D tensor where each row is a channel
        # and the columns are the mean and variance
        channel_stats = torch.stack((channel_means, channel_vars), dim=1)

        # Perform constrained k-means clustering on the channel statistics
        kmeans = KMeansConstrained(n_clusters=self.num_groups,
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

    def plot_groups(self, channels_groups, means, vars):
        groups = torch.repeat_interleave(torch.arange(self.num_groups),
                                         len(channels_groups) // self.num_groups,
                                         dim=0)
        # groups = np.repeat(np.arange(self.num_groups),
        #                    len(channels_groups) / self.num_groups)

        # Create a scatter plot with points colored by group
        plt.figure(figsize=(10, 10))
        plt.scatter(means[channels_groups].detach().numpy(),
                    vars[channels_groups].detach().numpy(), c=groups,
                    cmap='tab20', alpha=0.5)

        # Optionally, add a colorbar
        plt.colorbar(label='Group')

        plt.show()
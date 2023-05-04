import numpy as np
import torch
from torch import clone, empty_like, tensor, sort, zeros_like, cat, ceil, floor
from torch.nn import Module, GroupNorm
import global_vars
from agn_utils import getLayerIndex
import heapq


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
        N, C, W, H = Conv_input.size()
        self.indexes = self.SimilarityGroupNormClustering(clone(Conv_input),
                                                          self.num_groups).to(
            dtype=torch.int64)

        self.reverse_indexes = empty_like(self.indexes)
        for ind in range(N * C):
            self.reverse_indexes[self.indexes[ind]] = tensor(ind,
                                                             device=global_vars.device)

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

        else:
            print(
                f"SGN_version number {global_vars.args.SGN_version} is not available!")
            exit(1)

        #self.get_groups_representation_num(channels_input, channelsClustering)
        self.get_channels_clustering_for_eval(channels_input,
                                            channelsClustering)

        return channelsClustering

    def harmonic_mean(self, _tensor):
        """
        Calculate the harmonic mean of 2 tensors, for the var and mean of
        the tensor.
        """
        if _tensor.dim() == 4:
            N, C, H, W = _tensor.size()
            input_NC_WH = _tensor.reshape(N * C, H * W)

            mean = input_NC_WH.mean(dim=1)
            var = input_NC_WH.var(dim=1)

        elif _tensor.dim() == 2:
            mean = _tensor.mean()
            var = _tensor.var()
        else:
            raise Exception('harmonic mean support 4 or 2 dim only')

        return 2 * (mean * var) / (mean + var)

    def get_channels_clustering_for_eval(self, channels_input: torch.Tensor,
                                         channelsClustering: torch.Tensor):
        N, C, _, _ = channels_input.size()
        channel_groups = {i: [] for i in range(C)}
        for i in range(N*C):
            channel_to = channelsClustering[i] % C
            group_num = self.map_to_group(C, channel_to)
            channel_from = i % C
            channel_groups[channel_from].append(group_num)

        max_elements = self.get_num_occurrences(channel_groups)

        final_channel_groups = {i: [] for i in range(self.num_groups)}

        # Create a max heap (using negative values)
        min_heap = [(-max_val, i, group) for i, max_vals in
                    enumerate(max_elements.values()) for group, max_val in
                    enumerate(max_vals)]

        # Heapify the max heap
        heapq.heapify(min_heap)
        added_indices = set()
        while True:
            # Get the corresponding index and group of the maximum value
            neg_max_value, i, group = heapq.heappop(min_heap)

            if len(final_channel_groups[group]) < self.group_size and i not in added_indices:
                final_channel_groups[group].append(i)
                added_indices.add(i)

            # Break the loop when all groups are filled
            if all(len(lst) == self.group_size for lst in
                   final_channel_groups.values()):
                break

        # Flatten the lists in the dictionary
        flat_list = [elem for lst in final_channel_groups.values() for elem in lst]

        # Convert the flattened list to a PyTorch tensor
        factors = torch.arange(0, N) * C

        eval_indexes = torch.tensor(flat_list)
        eval_indexes = \
            eval_indexes.repeat(N).reshape(N, C) + factors.unsqueeze(1)
        self.eval_indexes = eval_indexes.reshape(-1).to(
                channels_input.device)

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
        num_counts = {i: [0]*self.num_groups for i in range(len(d))}

        for i, lst in enumerate(d.values()):
            for num in lst:
                num_counts[i][num] += 1

        return num_counts

    def map_to_group(self, C, X):
        group_size = C // self.num_groups
        group_num = X // group_size
        return group_num.item()

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
                channel_reps = torch.empty(self.num_channels, device=tensor.device)
                for i in range(self.num_channels):
                    channel_reps[i] = metric(tensor[b, i, :, :]).item()

                # Compute the distance between each channel and all representative numbers
                distances = torch.abs(channel_reps.view(-1, 1) - torch.tensor(self.groups_representation_num, device=tensor.device))

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

        return channelsClustering

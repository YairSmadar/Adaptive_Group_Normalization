from similarity_group_normalization import SimilarityGroupNorm
from random_group_normalization import RandomGroupNorm
import torch.nn as nn


class AGNScheduler:
    def __init__(self, model: nn.Module, epoch_start_cluster: int = 10,
                 num_of_epch_to_shuffle: int = 20,
                 riar: int = 1, max_norm_shuffle: int = 100):
        self.model = model
        self.epoch_num = -1
        self.num_of_epch_to_shuffle = num_of_epch_to_shuffle
        self.riar = riar
        self.max_norm_shuffle = max_norm_shuffle
        self.epoch_start_cluster = epoch_start_cluster
        self.recluster = False
        self.use_gn = False

    def step(self):
        self.update_recluster()
        for module in self.model.modules():
            if isinstance(module, (SimilarityGroupNorm, RandomGroupNorm)):
                module.need_to_recluster = self.recluster
                module.use_gn = self.use_gn

    def update_recluster(self):

        self.epoch_num += 1

        # update use_gn flag
        self.use_gn = self.epoch_start_cluster > self.epoch_num

        # update recluster flag
        if not self.model.training:
            self.recluster = False
            return

        shifted_epoch = self.epoch_num - self.epoch_start_cluster

        recluster_gap_pass = (shifted_epoch % self.num_of_epch_to_shuffle) in range(self.riar)

        self.recluster = recluster_gap_pass and \
                            (self.max_norm_shuffle > self.epoch_num >= self.epoch_start_cluster)

        if self.recluster:
            print(f'recluster at epoch {self.epoch_num}!')

        if self.use_gn:
            print(f'use_gn at epoch {self.epoch_num}!')

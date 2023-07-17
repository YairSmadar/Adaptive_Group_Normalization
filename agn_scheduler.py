from similarity_group_normalization import SimilarityGroupNorm
from random_group_normalization import RandomGroupNorm
import torch.nn as nn


class AGNScheduler:
    def __init__(self, model: nn.Module, epoch_start_cluster: int = 10,
                 num_of_epch_to_shuffle: int = 20,
                 riar: int = 1, max_norm_shuffle: int = 100):
        self.model = model
        self.recluster = False
        self.epoch_num = -1
        self.num_of_epch_to_shuffle = num_of_epch_to_shuffle
        self.riar = riar
        self.max_norm_shuffle = max_norm_shuffle
        self.epoch_start_cluster = epoch_start_cluster

    def step(self):
        self.update_recluster()
        for module in self.model.modules():
            if isinstance(module, (SimilarityGroupNorm, RandomGroupNorm)):
                module.need_to_recluster = self.recluster

    def update_recluster(self):

        if not self.model.training:
            return False

        self.epoch_num += 1

        epoch_clustring_loop = self.epoch_num % self.num_of_epch_to_shuffle

        self.recluster = (epoch_clustring_loop < self.riar) and \
                         self.max_norm_shuffle > self.epoch_num >= self.epoch_start_cluster

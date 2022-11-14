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
        if global_vars.recluster:
            self.recluster()
        return self.groupNorm(Conv_input[:, self.indexes, :, :])[:, self.reverse_indexes, :, :].requires_grad_(
            requires_grad=True)

    def recluster(self):
        self.indexes = sort(randperm(self.num_channels, generator=global_vars.generator))[1]
        #         self.indexes = better_places(self.indexes, self.num_groups, self.group_size, self.num_channels)
        self.reverse_indexes = empty_like(self.indexes, device=global_vars.device)
        for ind in range(self.num_channels):
            self.reverse_indexes[self.indexes[ind]] = tensor(ind, device=global_vars.device)


def better_places(indexes, num_groups, group_size, num_channels):
    a = indexes.reshape(num_groups, -1).to(device=global_vars.device)
    b = div(a, group_size, rounding_mode='floor')
    mat = []
    for j in range(num_groups):
        b0 = zeros(num_groups)
        for l in range(group_size):
            b0[b[j][l]] += 1
        c = b0.clone().detach().requires_grad_(False)
        mat.append(c)
    w = cat(mat, dim=-1).reshape(-1, num_groups)
    _max = max(w, dim=1)
    _sorted = sort(_max.values, descending=True)
    b2 = empty_like(b).to(device=global_vars.device)
    a2 = empty_like(a).to(device=global_vars.device)
    w2 = w.clone().to(device=global_vars.device)
    been, unbeen = [], []

    for ind in _sorted.indices:
        argmax = int(argmax(w2[ind]))
        if argmax in been:
            unbeen.append(int(ind))
        else:
            been.append(argmax)
            b2[argmax] = b[int(ind)]
            a2[argmax] = a[int(ind)]
            w2[:, argmax] = 0

    for i in range(num_groups):
        if not i in been:
            b2[i] = b[int(unbeen[0])]
            a2[i] = a[int(unbeen[0])]
            unbeen[1:]

    a3 = a2.clone().to(device=global_vars.device)
    fixd = arange(num_channels).reshape(num_groups, -1).to(device=global_vars.device)
    for i in range(num_groups):
        for j, k in zip(fixd[i], range(group_size)):
            if j in a2[i]:
                ind = int(np.argwhere(a2[i] == j)[0, 0])
                a3[i, [ind, k]] = a2[i, [k, ind]]
    return a3.reshape(-1).to(device=global_vars.device)

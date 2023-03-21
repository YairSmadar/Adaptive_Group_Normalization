from torch.nn import BatchNorm2d, GroupNorm, LayerNorm

import global_vars
from global_vars import groupsBySize
from random_group_normalization import RandomGroupNorm as rgn
from similarity_group_normalization import SimilarityGroupNorm as sgn


def norm2d(planes):
    if global_vars.args.method == "BN":
        return BatchNorm2d(planes)
    if global_vars.args.group_by_size:
        if global_vars.args.group_norm_size >= planes:
            return LayerNorm(planes, eps=global_vars.args.eps)
        numofgroups = int(groupsBySize(planes))
        if global_vars.args.method == "GN":
            return GroupNorm(numofgroups, planes, eps=global_vars.args.eps)
        elif global_vars.args.method == "RGN":
            return rgn(numofgroups, planes, eps=global_vars.args.eps)
        elif global_vars.args.method == "SGN":
            return sgn(numofgroups, planes, eps=global_vars.args.eps)
        else:
            raise Exception("the normalization method not recognized")
    if global_vars.args.method == "GN":
        return GroupNorm(global_vars.args.group_norm, planes, eps=global_vars.args.eps)
    elif global_vars.args.method == "RGN":
        return rgn(global_vars.args.group_norm, planes, eps=global_vars.args.eps)
    elif global_vars.args.method == "SGN":
        return sgn(global_vars.args.group_norm, planes, eps=global_vars.args.eps)
    else:
        raise Exception("the normalization method not recognized")

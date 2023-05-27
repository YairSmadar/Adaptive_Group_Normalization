from torch.nn import BatchNorm2d, GroupNorm, LayerNorm

import global_vars
from global_vars import groupsBySize
from random_group_normalization import RandomGroupNorm as rgn
from similarity_group_normalization import SimilarityGroupNorm as sgn
import similarity_group_normalization


def create_strategy(SGN_version, groups_num, ch_num):
    strategy_class_name = f'SortChannelsV{SGN_version}'
    strategy_class = getattr(similarity_group_normalization,
                             strategy_class_name, None)

    if strategy_class is None:
        print(f"SGN_version number {SGN_version} is not available!")
        exit(1)

    return strategy_class(groups_num, ch_num)


def norm2d(planes):
    if global_vars.args.method == "BN":
        return BatchNorm2d(planes)
    group_norm = global_vars.args.group_norm
    SGN_version = global_vars.args.SGN_version
    if global_vars.args.group_by_size:
        if global_vars.args.group_norm_size >= planes:
            return LayerNorm(planes, eps=global_vars.args.eps)
        numofgroups = int(groupsBySize(planes))
        if global_vars.args.method == "GN":
            return GroupNorm(numofgroups, planes, eps=global_vars.args.eps)
        elif global_vars.args.method == "RGN":
            return rgn(numofgroups, planes, eps=global_vars.args.eps)
        elif global_vars.args.method == "SGN":
            return sgn(numofgroups, planes, eps=global_vars.args.eps,
                       strategy=create_strategy(SGN_version, numofgroups,
                                                planes))
        else:
            raise Exception("the normalization method not recognized")
    if global_vars.args.method == "GN":
        return GroupNorm(group_norm, planes, eps=global_vars.args.eps)
    elif global_vars.args.method == "RGN":
        return rgn(group_norm, planes, eps=global_vars.args.eps)
    elif global_vars.args.method == "SGN":
        return sgn(group_norm, planes, eps=global_vars.args.eps,
                   strategy=create_strategy(SGN_version, group_norm,
                                            planes))
    else:
        raise Exception("the normalization method not recognized")

from torch.nn import BatchNorm2d, GroupNorm, LayerNorm

import global_vars
from global_vars import groupsBySize
from random_group_normalization import RandomGroupNorm as rgn
from similarity_group_normalization import SimilarityGroupNorm as sgn


def norm2d(planes):
    args = global_vars.args

    if args.method == "BN":
        return BatchNorm2d(planes)

    if args.group_by_size:
        if args.group_norm_size >= planes:
            return LayerNorm(planes, eps=args.eps)

        num_of_groups = int(groupsBySize(planes))

        if args.method == "GN":
            return GroupNorm(num_of_groups, planes, eps=args.eps)

        elif args.method == "RGN":
            return rgn(num_of_groups, planes, eps=args.eps)

        elif args.method == "SGN":
            return sgn(num_of_groups, planes, eps=args.eps)

        else:
            raise Exception("the normalization method not recognized")

    if args.method == "GN":
        return GroupNorm(args.group_norm, planes, eps=args.eps)

    elif args.method == "RGN":
        return rgn(args.group_norm, planes, eps=args.eps)

    elif args.method == "SGN":
        return sgn(args.group_norm, planes, eps=args.eps)

    else:
        raise Exception("the normalization method not recognized")

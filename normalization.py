from torch.nn import BatchNorm2d, GroupNorm, LayerNorm
from random_group_normalization import RandomGroupNorm as rgn
from similarity_group_normalization import SimilarityGroupNorm as sgn
import similarity_group_normalization


class NormalizationFactory:
    def __init__(self, normalization_args):
        self.normalization_args = normalization_args

    def create_strategy(self, group_norm, planes):
        strategy_class_name = f'SortChannelsV{self.normalization_args["version"]}'
        strategy_class = getattr(similarity_group_normalization, strategy_class_name, None)

        if strategy_class is None:
            raise Exception(f"{self.normalization_args['method']} version number "
                            f"{self.normalization_args['version']} is not available!")

        return strategy_class(group_norm, planes, self.normalization_args)

    def init_norm_layer(self, norm_layer):
        if isinstance(norm_layer, (BatchNorm2d, GroupNorm)):
            norm_layer.weight.data.fill_(1)
            norm_layer.bias.data.zero_()
        elif isinstance(norm_layer, rgn):
            norm_layer.groupNorm.weight.data.fill_(1)
            norm_layer.groupNorm.bias.data.zero_()
        elif isinstance(norm_layer, sgn):
            norm_layer.groupNorm.weight.data.fill_(1)
            norm_layer.groupNorm.bias.data.zero_()
        else:
            raise Exception("Normalization layer not recognized for initialization")

    def groupsBySize(self, numofchannels):
        return int(numofchannels / self.normalization_args["group_size"])

    def create_norm2d(self, planes):
        method = self.normalization_args["method"]
        if method == "BN":
            normalization_layer = BatchNorm2d(planes)
        else:
            group_norm = self.normalization_args["group_norm"]
            eps = self.normalization_args["eps"]
            if self.normalization_args["group_by_size"]:
                if self.normalization_args["group_size"] >= planes:
                    normalization_layer = LayerNorm(planes, eps=eps)
                else:
                    numofgroups = self.groupsBySize(planes)
                    if method == "GN":
                        normalization_layer = GroupNorm(numofgroups, planes, eps=eps)
                    elif method == "RGN":
                        normalization_layer = rgn(num_groups=numofgroups, num_channels=planes, eps=eps)
                    elif method == "SGN":
                        normalization_layer = sgn(num_groups=numofgroups, num_channels=planes, eps=eps,
                                                  strategy=self.create_strategy(numofgroups, planes),
                                                  normalization_args=self.normalization_args)
                    else:
                        raise Exception("The normalization method not recognized")
            else:
                if method == "GN":
                    normalization_layer = GroupNorm(group_norm, planes, eps=eps)
                elif method == "RGN":
                    normalization_layer = rgn(num_groups=group_norm, num_channels=planes, eps=eps)
                elif method == "SGN":
                    normalization_layer = sgn(num_groups=group_norm, num_channels=planes, eps=eps,
                                              strategy=self.create_strategy(group_norm, planes),
                                              normalization_args=self.normalization_args)
                else:
                    raise Exception("The normalization method not recognized")

        # After you create the normalization_layer, initialize it
        self.init_norm_layer(normalization_layer)

        return normalization_layer

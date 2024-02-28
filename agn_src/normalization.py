from torch.nn import BatchNorm2d, LayerNorm
if False:
    from agn_src.GroupNormMyImpl import GroupNormMyImpl as GroupNorm
else:
    from torch.nn import GroupNorm
from agn_src.random_group_normalization import RandomGroupNorm as rgn
from agn_src.similarity_group_normalization import SimilarityGroupNorm as sgn
import agn_src.similarity_group_normalization as similarity_group_normalization


class NormalizationFactory:
    def __init__(self,
                 version: int = 14,
                 method: str = "GN", group_by_size: bool = True,
                 group_norm_size: int = 32, group_norm: int = 16,
                 plot_groups: bool = False,
                 eps: float = 1e-12, no_shuff_best_k_p: float = 1.0,
                 shuff_thrs_std_only: bool = False,
                 std_threshold_l: int = 0, std_threshold_h: int = 1,
                 keep_best_group_num_start: int = 0,
                 use_VGN: bool = False,
                 VGN_min_gs_mul: float = 1,
                 VGN_max_gs_mul: float = 1
                 ):
        self.version = version

        # factory args
        self.method = method
        self.group_by_size = group_by_size
        self.group_size = group_norm_size if group_by_size else group_norm
        self.group_norm = group_norm
        self.plot_groups = plot_groups

        # sgn args
        self.eps = eps
        self.no_shuff_best_k_p = no_shuff_best_k_p
        self.shuff_thrs_std_only = shuff_thrs_std_only
        self.std_threshold_l = std_threshold_l
        self.std_threshold_h = std_threshold_h
        self.keep_best_group_num_start = keep_best_group_num_start
        self.use_VGN = use_VGN
        self.VGN_min_gs_mul = VGN_min_gs_mul
        self.VGN_max_gs_mul = VGN_max_gs_mul

    def create_strategy(self, group_norm, planes):
        strategy_class_name = f'SortChannelsV{self.version}'
        strategy_class = getattr(similarity_group_normalization, strategy_class_name, None)

        if strategy_class is None:
            raise Exception(f"{self.method} version number "
                            f"{self.version} is not available!")

        return strategy_class(num_groups=group_norm, num_channels=planes,
                              plot_groups=self.plot_groups, use_VGN=self.use_VGN,
                              VGN_min_gs_mul=self.VGN_min_gs_mul, VGN_max_gs_mul=self.VGN_max_gs_mul)

    def init_norm_layer(self, norm_layer):
        if isinstance(norm_layer, (BatchNorm2d, GroupNorm, LayerNorm)):
            norm_layer.weight.data.fill_(1)
            norm_layer.bias.data.zero_()
        elif isinstance(norm_layer, (sgn, rgn)):
            norm_layer.groupNorm.weight.data.fill_(1)
            norm_layer.groupNorm.bias.data.zero_()
        else:
            raise Exception("Normalization layer not recognized for initialization")

    def groupsBySize(self, numofchannels):
        return self.find_closest_natural_divisor(numofchannels, self.group_size)

    def find_closest_natural_divisor(self, x, y):
        m = x / y  # Calculate M
        m = min(m, int(m))  # Ensure M is the minimum of M and int(M)

        # if m == 0:
        #     return 1  # one group
        # If X divided by M is already a natural number, and M is an integer, return M
        if m is not 0 and x % m == 0:
            return int(m)  # Return M as an integer

        closest_divisor = None
        smallest_difference = float('inf')  # Initialize with an infinitely large value

        # Iterate over possible divisors from 1 to X
        for i in range(1, x + 1):
            if x % i == 0:  # Check if i is a divisor of X
                difference = abs(m - i)  # Calculate the difference from M
                if difference < smallest_difference:  # Check if this is the smallest difference so far
                    closest_divisor = i
                    smallest_difference = difference

        return closest_divisor

    def create_norm2d(self, planes):
        normalization_layer_dict = {
            "BN": BatchNorm2d,
            "GN": GroupNorm,
            "SGN": sgn,
            "RGN": rgn
        }

        if self.method not in normalization_layer_dict:
            raise Exception("The normalization method not recognized")

        if self.method == "BN":
            normalization_layer = normalization_layer_dict[self.method](num_features=planes, eps=self.eps)
        else:
            num_groups = self.groupsBySize(planes) if self.group_by_size else self.group_norm

            # if self.group_by_size and self.group_size >= planes:
            #     normalization_layer = LayerNorm(planes, eps=self.eps)
            if self.method == "GN":
                normalization_layer = GroupNorm(num_groups, planes, eps=self.eps)
            elif self.method == "SGN":
                normalization_layer = normalization_layer_dict[self.method](
                    num_groups=num_groups, num_channels=planes,
                    strategy=self.create_strategy(num_groups, planes),
                    version=self.version, eps=self.eps,
                    no_shuff_best_k_p=self.no_shuff_best_k_p,
                    shuff_thrs_std_only=self.shuff_thrs_std_only,
                    std_threshold_l=self.std_threshold_l,
                    std_threshold_h=self.std_threshold_h,
                    keep_best_group_num_start=self.keep_best_group_num_start,
                    use_VGN=self.use_VGN
                )
            else:
                normalization_layer = normalization_layer_dict[self.method](
                    num_groups=num_groups, num_channels=planes, eps=self.eps
                )

        # After you create the normalization_layer, initialize it
        self.init_norm_layer(normalization_layer)

        return normalization_layer


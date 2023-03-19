import math
import torch
import torch.nn as nn
from torch.nn.modules import normalization

import global_vars
from normalization import norm2d
from random_group_normalization import RandomGroupNorm as rgn
from similarity_group_normalization import SimilarityGroupNorm as sgn


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        global flag
        self.method = global_vars.args.method
        self.group_norm = global_vars.args.group_norm
        self.inplanes = 64
        self.normLayers = []
        self.batch_num = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, bias=False)
        self.norm1 = norm2d(64)
        # self.norm256 = norm2d(256)
        # self.norm512 = norm2d(512)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ac_gn=global_vars.args.GN_in_bt)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ac_gn=global_vars.args.GN_in_bt)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ac_gn=global_vars.args.GN_in_bt)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, rgn):
                m.groupNorm.weight.data.fill_(1)
                m.groupNorm.bias.data.zero_()
            elif isinstance(m, sgn):
                m.groupNorm.weight.data.fill_(1)
                m.groupNorm.bias.data.zero_()
            elif isinstance(m, Bottleneck):
                n1 = m.conv1.kernel_size[0] * m.conv1.kernel_size[1] * m.conv1.out_channels
                m.conv1.weight.data.normal_(0, math.sqrt(2. / n1))
                n2 = m.conv2.kernel_size[0] * m.conv2.kernel_size[1] * m.conv2.out_channels
                m.conv2.weight.data.normal_(0, math.sqrt(2. / n2))
                n3 = m.conv3.kernel_size[0] * m.conv3.kernel_size[1] * m.conv3.out_channels
                m.conv3.weight.data.normal_(0, math.sqrt(2. / n3))

                try:
                    if self.method == 'SGN' or self.method == 'RGN':
                        m.norm1.groupNorm.weight.data.fill_(1)
                        m.norm1.groupNorm.bias.data.zero_()
                        m.norm2.groupNorm.weight.data.fill_(1)
                        m.norm2.groupNorm.bias.data.zero_()
                        m.norm3.groupNorm.weight.data.fill_(1)
                        m.norm3.groupNorm.bias.data.zero_()
                    else:
                        m.norm1.weight.data.fill_(1)
                        m.norm1.bias.data.zero_()
                        m.norm2.weight.data.fill_(1)
                        m.norm2.bias.data.zero_()
                        m.norm3.weight.data.fill_(1)
                        m.norm3.bias.data.zero_()
                except:
                    m.norm1.weight.data.fill_(1)
                    m.norm1.bias.data.zero_()
                    m.norm2.weight.data.fill_(1)
                    m.norm2.bias.data.zero_()
                    m.norm3.weight.data.fill_(1)
                    m.norm3.bias.data.zero_()

        # self.normLayers.append(self.norm1)
        # # for normLayer in [self.layer1, self.layer2, self.layer3, self.layer4]:
        # self.normLayers.append(self.layer1.named_modules['0'])#norm1)
        #   # self.normLayers.append(normLayer.norm2)
        #   # self.normLayers.append(normLayer.norm3)

    def _make_layer(self, block, planes, blocks, stride=1, ac_gn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm2d(planes * block.expansion, self.batch_num),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.group_norm, self.method, stride, downsample, ac_gn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.group_norm, self.method))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        if global_vars.is_agn:
            x = self.norm1(x, self.batch_num)
        else:
            x = self.norm1(x)

        x = self.relu(x)
        x = self.layer1(x)

        # if global_vars.is_agn:
        #     x = self.norm256(x, self.batch_num)
        # else:
        #     x = self.norm256(x)

        x = self.layer2(x)

        # if global_vars.is_agn:
        #     x = self.norm512(x, self.batch_num)
        # else:
        #     x = self.norm512(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, group_norm, method, stride=1, downsample=None, ac_gn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = norm2d(planes, ac_gn)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = norm2d(planes, ac_gn)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.norm3 = norm2d(planes * 4, ac_gn)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)

        if global_vars.is_agn:
            out = self.norm1(out, global_vars.batch_num)
        else:
            out = self.norm1(out)

        out = self.relu(out)
        out = self.conv2(out)

        if global_vars.is_agn:
            out = self.norm2(out, global_vars.batch_num)
        else:
            out = self.norm2(out)

        out = self.relu(out)
        out = self.conv3(out)

        if global_vars.is_agn:
            out = self.norm3(out, global_vars.batch_num)
        else:
            out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

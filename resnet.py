import math
import torch.nn as nn

import global_vars
from normalization import norm2d
from random_group_normalization import RandomGroupNorm as rgn
from similarity_group_normalization import SimilarityGroupNorm as sgn


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   dropout_p=global_vars.args.dropout_prop)
    return model


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, dropout_p=0.2):
        super(ResNet, self).__init__()
        self.method = global_vars.args.method
        self.group_norm = global_vars.args.group_norm
        self.inplanes = 64
        self.normLayers = []
        self.batch_num = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, bias=False)
        self.norm1 = norm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.dropout_p = dropout_p
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.dropout = nn.Dropout(dropout_p)
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

                if global_vars.args.SGN_version == 17:
                    for inst in m.variableGroupNorm.instance_norms:
                        inst.weight.data.fill_(1)
                        inst.bias.data.zero_()

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

                        if global_vars.args.SGN_version == 17:
                            for inst in m.norm1.variableGroupNorm.instance_norms:
                                inst.weight.data.fill_(1)
                                inst.bias.data.zero_()
                            for inst in m.norm2.variableGroupNorm.instance_norms:
                                inst.weight.data.fill_(1)
                                inst.bias.data.zero_()
                            for inst in m.norm3.variableGroupNorm.instance_norms:
                                inst.weight.data.fill_(1)
                                inst.bias.data.zero_()


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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm2d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, self.group_norm, self.method, stride, downsample, dropout_p=self.dropout_p))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.group_norm, self.method))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if global_vars.args.model_version == 3:
            x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if global_vars.args.model_version == 2:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, group_norm, method, stride=1, downsample=None, dropout_p=0.5):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = norm2d(planes)
        self.dropout1 = nn.Dropout(dropout_p)  # Dropout after norm1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = norm2d(planes)
        self.dropout2 = nn.Dropout(dropout_p)  # Dropout after norm2
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.norm3 = norm2d(planes * 4)
        self.dropout3 = nn.Dropout(dropout_p)  # Dropout after norm3
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        if global_vars.args.model_version == 3:
            out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        if global_vars.args.model_version == 3:
            out = self.dropout2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        if global_vars.args.model_version == 3:
            out = self.dropout3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


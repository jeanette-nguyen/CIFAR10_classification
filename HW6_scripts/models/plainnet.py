import torch.nn as nn
import math

__all__ = ['PlainNet', 'plainnet8', 'plainnet20', 'plainnet56']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        in_bn1 = self.conv1(x)
        out = self.bn1(in_bn1)
        out = self.relu(out)

        in_bn2 = self.conv2(out)
        out = self.bn2(in_bn2)
        out = self.relu(out)

        return out, in_bn1, in_bn2


class PlainNet(nn.Module):

    def __init__(self, block, layers, inplanes=16, num_classes=10):
        self.inplanes = inplanes
        super(PlainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(inplanes * 4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        in_bn1 = self.conv1(x)
        x = self.bn1(in_bn1)
        x = self.relu(x)

        x, in_bn2, in_bn3 = self.layer1(x)
        x, in_bn4, in_bn5  = self.layer2(x)
        x, in_bn6, in_bn7  = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, in_bn1, in_bn3, in_bn5, in_bn7



def plainnet20(inplanes, num_classes):
    model = PlainNet(BasicBlock, [3, 3, 3], inplanes, num_classes)
    return model


def plainnet56(inplanes, num_classes):
    model = PlainNet(BasicBlock, [9, 9, 9], inplanes, num_classes)
    return model

def plainnet8(inplanes, num_classes):
    model = PlainNet(BasicBlock, [1,1,1], inplanes, num_classes)
    return model

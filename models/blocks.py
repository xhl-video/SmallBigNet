import torch

import torch.nn as nn


class Bottleneck3D_100(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            t_stride=1,
            downsample=None,
            t_length=None):
        super(Bottleneck3D_100, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes,
            kernel_size=(3, 1, 1),
            stride=(t_stride, 1, 1),
            padding=(1, 0, 0),
            bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes,
            planes *
            self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D_000(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            t_stride=1,
            downsample=None,
            t_length=None):
        super(Bottleneck3D_000, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes,
            kernel_size=1,
            stride=[t_stride, 1, 1],
            bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=(1, 3, 3),
            stride=[1, stride, stride],
            padding=(0, 1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck3D_100_extra(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            t_stride=1,
            downsample=None,
            t_length=None):
        super(Bottleneck3D_100_extra, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes,
            kernel_size=(3, 1, 1),
            stride=(t_stride, 1, 1),
            padding=(1, 0, 0),
            bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes,
            planes *
            self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.big4_1 = nn.Conv3d(
            inplanes,
            inplanes // 4,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False)
        self.big4_2 = nn.Sequential(nn.Conv3d(
            inplanes // 4,
            inplanes,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False),
            nn.BatchNorm3d(inplanes))
        self.big_extra = nn.AdaptiveMaxPool3d(1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = x + self.big4_2(self.big4_1(x) * self.sigmod(self.big4_1(self.big_extra(x))))
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SmallBig_module(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            t_stride=1,
            t_length=8,
            downsample=None):
        super(SmallBig_module, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes,
            kernel_size=1,
            stride=[t_stride, 1, 1],
            bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=(1, 3, 3),
            stride=[1, stride, stride],
            padding=(0, 1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes,
            planes *
            self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.big1 = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=3,
                padding=1,
                stride=1),
            nn.BatchNorm3d(planes))

        self.big2 = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=3,
                padding=1,
                stride=1),
            nn.BatchNorm3d(planes))

        self.big3 = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=(t_length, 3, 3),
                padding=(0, 1, 1),
                stride=1),
            nn.BatchNorm3d(planes * self.expansion))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out + self.big1[1](self.conv1(self.big1[0](x))))
        out = self.relu(self.bn2(self.conv2(out)) + self.big2[1](self.conv2(self.big2[0](out))))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.bn3(self.conv3(out)) + self.big3[1](self.conv3(self.big3[0](out)))
        out += residual
        out = self.relu(out)

        return out

class SmallBig_module_extra(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            t_stride=1,
            t_length=8,
            downsample=None):
        super(SmallBig_module_extra, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes,
            kernel_size=1,
            stride=[t_stride, 1, 1],
            bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=(1, 3, 3),
            stride=[1, stride, stride],
            padding=(0, 1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes,
            planes *
            self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.big1 = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=3,
                padding=1,
                stride=1),
            nn.BatchNorm3d(planes))

        self.big2 = nn.Sequential(
            nn.MaxPool3d(
                kernel_size=3,
                padding=1,
                stride=1),
            nn.BatchNorm3d(planes))

        self.big3 = nn.Sequential(
                nn.MaxPool3d(
                    kernel_size=(t_length, 3, 3),
                    padding=(0, 1, 1),
                    stride=1),
                nn.BatchNorm3d(planes * self.expansion))
        self.downsample = downsample
        self.stride = stride
        self.big4_1 = nn.Conv3d(
            inplanes,
            inplanes // 4,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False)
        self.big4_2 = nn.Sequential(nn.Conv3d(
            inplanes // 4,
            inplanes,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False),
            nn.BatchNorm3d(inplanes))
        self.big_extra = nn.AdaptiveMaxPool3d(1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):

        x = x + self.big4_2(self.big4_1(x) * self.sigmod(self.big4_1(self.big_extra(x))))

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out + self.big1[1](self.conv1(self.big1[0](x))))
        out = self.relu(self.bn2(self.conv2(out)) + self.big2[1](self.conv2(self.big2[0](out))))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.bn3(self.conv3(out)) + self.big3[1](self.conv3(self.big3[0](out)))
        out += residual
        out = self.relu(out)

        return out


class SmallBig_plus_module_extra(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            t_stride=1,
            t_length=8,
            downsample=None):
        super(SmallBig_plus_module_extra, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes,
            kernel_size=1,
            stride=[t_stride, 1, 1],
            bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=(1, 3, 3),
            stride=[1, stride, stride],
            padding=(0, 1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes,
            planes *
            self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.big1 = nn.Sequential(
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
                padding=0,
                stride=(1, 2, 2)),
            nn.BatchNorm3d(planes),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.big2 = nn.Sequential(
            nn.AvgPool3d(
                kernel_size=(1, 2, 2),
                padding=0,
                stride=(1, 2, 2)),
            nn.BatchNorm3d(planes),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.big3 = nn.Sequential(
            nn.AvgPool3d(
                kernel_size=(t_length, 2, 2),
                padding=0,
                stride=(1, 2, 2)),
            nn.BatchNorm3d(planes),
            nn.Upsample(scale_factor=(1, 2, 2)),
        )
        self.downsample = downsample
        self.stride = stride
        self.big4_1 = nn.Conv3d(
            inplanes,
            inplanes // 4,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False)
        self.big4_2 = nn.Sequential(nn.Conv3d(
            inplanes // 4,
            inplanes,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False),
            nn.BatchNorm3d(inplanes))
        self.big_extra = nn.AdaptiveMaxPool3d(1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):

        x = x + self.big4_2(self.big4_1(x) * self.sigmod(self.big4_1(self.big_extra(x))))

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out +  self.big1[2](self.big1[1](self.conv1(self.big1[0](x)))))
        out = self.relu(self.bn2(self.conv2(out)) +  self.big2[2](self.big2[1](self.conv2(self.big2[0](out)))))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.bn3(self.conv3(out)) + self.big3[2](self.big3[1](self.conv3(self.big3[0](out))))
        out += residual
        out = self.relu(out)

        return out

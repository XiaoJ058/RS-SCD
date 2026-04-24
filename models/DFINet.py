"""
DFINet: Dual-Feature Interaction Network for Change Detection

Reference:
    Please add citation if applicable.

Author: [Your Name]
License: [e.g., MIT]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Parameter
from typing import List, Optional, Tuple
from utils.misc import initialize_weights
from thop import profile  # optional, for FLOPs calculation


# ------------------------------------------------------------------------------
# Basic building blocks
# ------------------------------------------------------------------------------

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicConv3D(nn.Module):
    """3D convolution block with optional batch norm and activation."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        bias: str = 'auto',
        bn: bool = False,
        act: bool = False,
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(nn.ConstantPad3d(kernel_size // 2, 0.0))
        seq.append(
            nn.Conv3d(
                in_ch, out_ch, kernel_size,
                padding=0,
                bias=(False if bn else True) if bias == 'auto' else bias,
                **kwargs
            )
        )
        if bn:
            seq.append(nn.GroupNorm(int(out_ch / 16), out_ch))
        if act:
            seq.append(nn.ReLU())
        self.seq = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Decompose_conv(nn.Module):
    """Decompose a 2D convolution into spatial and temporal 3D convolutions."""
    def __init__(
        self,
        conv2d: nn.Conv2d,
        time_dim: int = 3,
        time_padding: int = 0,
        time_stride: int = 1,
        time_dilation: int = 1,
        center: bool = False
    ):
        super().__init__()
        self.time_dim = time_dim
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])

        if time_dim == 1:
            self.conv3d = nn.Conv3d(
                conv2d.in_channels, conv2d.out_channels, kernel_dim,
                padding=padding, dilation=dilation, stride=stride
            )
            weight_2d = conv2d.weight.data
            weight_3d = weight_2d.unsqueeze(2)
            self.conv3d.weight = Parameter(weight_3d)
            self.conv3d.bias = conv2d.bias
        else:
            self.conv3d_spatial = nn.Conv3d(
                conv2d.in_channels, conv2d.out_channels,
                kernel_size=(1, kernel_dim[1], kernel_dim[2]),
                padding=(0, padding[1], padding[2]),
                dilation=(1, dilation[1], dilation[2]),
                stride=(1, stride[1], stride[2])
            )
            weight_2d = conv2d.weight.data
            self.conv3d_spatial.weight = Parameter(weight_2d.unsqueeze(2))
            self.conv3d_spatial.bias = conv2d.bias

            # temporal convolutions
            self.conv3d_time_1 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            self.conv3d_time_2 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            self.conv3d_time_3 = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, [1, 1, 1], bias=False)
            nn.init.constant_(self.conv3d_time_1.weight, 0.0)
            nn.init.constant_(self.conv3d_time_3.weight, 0.0)
            nn.init.eye_(self.conv3d_time_2.weight[:, :, 0, 0, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_dim == 1:
            return self.conv3d(x)
        else:
            x_spatial = self.conv3d_spatial(x)
            T1 = x_spatial[:, :, 0:1, :, :]
            T2 = x_spatial[:, :, 1:2, :, :]
            T1_F1 = self.conv3d_time_2(T1)
            T2_F1 = self.conv3d_time_2(T2)
            T1_F2 = self.conv3d_time_1(T1)
            T2_F2 = self.conv3d_time_3(T2)
            return torch.cat([T1_F1 + T2_F2, T1_F2 + T2_F1], dim=2)

    # ---------- Static helper methods ----------
    @staticmethod
    def Decompose_norm(batch2d: nn.BatchNorm2d) -> nn.BatchNorm3d:
        batch3d = nn.BatchNorm3d(batch2d.num_features)
        batch2d._check_input_dim = batch3d._check_input_dim
        return batch2d

    @staticmethod
    def Decompose_pool(
        pool2d,
        time_dim: int = 1,
        time_padding: int = 0,
        time_stride: Optional[int] = None,
        time_dilation: int = 1
    ):
        if isinstance(pool2d, nn.AdaptiveAvgPool2d):
            return nn.AdaptiveAvgPool3d((1, 1, 1))
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            return nn.MaxPool3d(kernel_dim, padding=padding, dilation=dilation, stride=stride,
                                ceil_mode=pool2d.ceil_mode)
        elif isinstance(pool2d, nn.AvgPool2d):
            return nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError(f'Unsupported pooling type: {type(pool2d)}')

    @staticmethod
    def inflate_conv(
        conv2d: nn.Conv2d,
        time_dim: int = 3,
        time_padding: int = 0,
        time_stride: int = 1,
        time_dilation: int = 1,
        center: bool = False
    ) -> nn.Conv3d:
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
        conv3d = nn.Conv3d(
            conv2d.in_channels, conv2d.out_channels, kernel_dim,
            padding=padding, dilation=dilation, stride=stride
        )
        weight_2d = conv2d.weight.data
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
        return conv3d

    @staticmethod
    def Decompose_layer(reslayer2d) -> nn.Sequential:
        layers = [Bottleneck3d(layer2d) for layer2d in reslayer2d]
        return nn.Sequential(*layers)

    @staticmethod
    def Decompose_downsample(downsample2d, time_stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            Decompose_conv.inflate_conv(downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
            Decompose_conv.Decompose_norm(downsample2d[1])
        )


class Bottleneck3d(nn.Module):
    """3D version of a bottleneck residual block."""
    def __init__(self, bottleneck2d: nn.Module):
        super().__init__()
        self.conv1 = Decompose_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = Decompose_conv.Decompose_norm(bottleneck2d.bn1)
        self.conv2 = Decompose_conv(bottleneck2d.conv2, time_dim=3, time_padding=1, time_stride=1, center=True)
        self.bn2 = Decompose_conv.Decompose_norm(bottleneck2d.bn2)
        self.conv3 = Decompose_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = Decompose_conv.Decompose_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)
        if bottleneck2d.downsample is not None:
            self.downsample = Decompose_conv.Decompose_downsample(bottleneck2d.downsample, time_stride=1)
        else:
            self.downsample = None
        self.stride = bottleneck2d.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ResBlock3D(nn.Module):
    """3D residual block with spatial/temporal decomposition."""
    def __init__(self, in_ch: int, out_ch: int, itm_ch: int, stride: int = 1, ds: Optional[nn.Module] = None):
        super().__init__()
        self.conv0 = nn.Conv2d(itm_ch, itm_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = BasicConv3D(in_ch, itm_ch, 1, bn=True, act=True, stride=stride)
        self.conv2 = Decompose_conv(self.conv0, time_dim=1, center=True)
        self.conv3 = BasicConv3D(itm_ch, out_ch, 1, bn=True, act=False)
        self.ds = ds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        return F.relu(y + res)


# ------------------------------------------------------------------------------
# Feature fusion and attention modules
# ------------------------------------------------------------------------------

class Light_Bag(nn.Module):
    """Light-weight bidirectional feature fusion for change decoder."""
    def __init__(self, in_channels: int, out_channels: int, BatchNorm: nn.Module = nn.BatchNorm2d):
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels)
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels)
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        edge_att = torch.sigmoid(d)
        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        return p_add + i_add


class CMA_variant(nn.Module):
    """Coordinate attention module variant (height + width pooling)."""
    def __init__(self, inp: int, oup: int, reduction: int = 1):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        return identity * x_h.expand(-1, -1, h, w) * x_w.expand(-1, -1, h, w)


class FF(nn.Module):
    """Feature fusion module with cross-level similarity."""
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        after_relu: bool = False,
        with_channel: bool = False,
        GroupNorm: nn.Module = nn.GroupNorm
    ):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.conv1 = conv1x1(3, 1)
        self.conv2 = conv1x1(128, 64)  # kept as original hard-coded logic
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            GroupNorm(int(mid_channels / 16), in_channels))
        self.CAM = CMA_variant(in_channels, in_channels)
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            GroupNorm(int(mid_channels / 16), mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            GroupNorm(int(mid_channels / 16), mid_channels)
        )
        self.f_z = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            GroupNorm(int(mid_channels / 16), mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                GroupNorm(int(in_channels / 16), in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
            z = self.relu(z)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=input_size[2:], mode='bilinear', align_corners=False)
        z_v = self.f_z(z)
        z_v = F.interpolate(z_v, size=input_size[2:], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            s1 = torch.sum(y_q * x_k, dim=1).unsqueeze(1)
            s2 = torch.sum(y_q * z_v, dim=1).unsqueeze(1)
            s3 = torch.sum(x_k * z_v, dim=1).unsqueeze(1)
            sim_map = torch.sigmoid(self.conv1(torch.cat((s1, s2, s3), dim=1)))

        y = F.interpolate(y, size=input_size[2:], mode='bilinear', align_corners=False)
        z = F.interpolate(z, size=input_size[2:], mode='bilinear', align_corners=False)
        x1 = self.conv2(torch.cat((y, z), dim=1))
        x = (1 - sim_map) * x + sim_map * x1
        x1 = self.head(x)
        x2 = self.CAM(x1)
        x = x + x2
        return x


# ------------------------------------------------------------------------------
# ASPP and spatial attention
# ------------------------------------------------------------------------------

class ASPPConv(nn.Sequential):
    """ASPP convolution module."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(int(out_channels / 16), out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    """ASPP global average pooling branch."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(int(out_channels / 16), out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(int(out_channels / 16), out_channels),
                nn.ReLU()
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(int(out_channels / 16), out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(_res, dim=1))


# ------------------------------------------------------------------------------
# Cross-temporal attention (CotSR)
# ------------------------------------------------------------------------------

class CotSR(nn.Module):
    """Cross-temporal feature refinement using dual attention."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv1 = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.ASPP = ASPP(128, [3, 9, 15], 128)
        self.conv1 = conv1x1(256, 128)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x1.size()
        q1 = self.query_conv1(x1).view(B, -1, H * W).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(B, -1, H * W)
        v1 = self.value_conv1(x1).view(B, -1, H * W)

        q2 = self.query_conv2(x2).view(B, -1, H * W).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(B, -1, H * W)
        v2 = self.value_conv2(x2).view(B, -1, H * W)

        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        attention = (attention1 + attention2) / 2

        out1 = torch.bmm(v2, attention.permute(0, 2, 1)).view(B, C, H, W)
        out2 = torch.bmm(v1, attention.permute(0, 2, 1)).view(B, C, H, W)
        out = self.gamma1 * out1 + self.gamma2 * out2

        out1 = x1 + out
        out2 = x2 + out

        x11 = self.ASPP(x1)
        x22 = self.ASPP(x2)
        out1 = self.conv1(torch.cat((x11, out1), dim=1))
        out2 = self.conv1(torch.cat((x22, out2), dim=1))
        return out1, out2


# ------------------------------------------------------------------------------
# 3D video temporal stem
# ------------------------------------------------------------------------------

class VTS(nn.Module):
    """Video Temporal Stem: extracts temporal features from paired frames."""
    def __init__(self, in_ch: int, enc_chs: int):
        super().__init__()
        self.expansion = 2
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, enc_chs, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(int(enc_chs / 16), enc_chs),
            nn.ReLU()
        )
        self.layer1 = ResBlock3D(in_ch, in_ch, in_ch)
        self.layer2 = ResBlock3D(enc_chs, enc_chs * self.expansion, enc_chs,
                                 ds=BasicConv3D(enc_chs, enc_chs * self.expansion, 1, bn=True))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.stem(x)
        return self.layer2(x)


# ------------------------------------------------------------------------------
# Backbone (FCN with ResNet34)
# ------------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Standard residual block for up-sampling path."""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(int(planes / 16), planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(int(planes / 16), planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class FCN(nn.Module):
    """Fully Convolutional Network with ResNet34 backbone."""
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()  # fixed missing parentheses
        resnet = models.resnet34(pretrained)

        # adjust input layer for custom channel number
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if in_channels >= 3:
            newconv1.weight.data[:, :3, :, :] = resnet.conv1.weight.data[:, :3, :, :]
            if in_channels > 3:
                newconv1.weight.data[:, 3:in_channels, :, :] = resnet.conv1.weight.data[:, :in_channels - 3, :, :]

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # keep spatial resolution in last two stages
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )
        initialize_weights(self.head)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))
        layers = [block(inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


# ------------------------------------------------------------------------------
# Main network: DFINet
# ------------------------------------------------------------------------------

class DFINet(nn.Module):
    """Dual-Feature Interaction Network for change detection."""
    def __init__(self, in_channels: int = 3, num_classes: int = 7):
        super().__init__()
        self.video_len = 4

        # backbone
        self.FCN = FCN(in_channels, pretrained=True)

        # spatial refinement
        self.SR = CotSR(128)

        # feature fusion
        self.F1 = FF(64, 128, after_relu=False)
        self.F2 = FF(64, 128, after_relu=False)

        # change decoder
        self.STC = Light_Bag(64, 128)
        self.Up1 = self._make_layer(ResBlock, 128, 64, 5, stride=1)
        self.Up2 = self._make_layer(ResBlock, 64, 64, 4, stride=1)
        self.Up3 = self._make_layer(ResBlock, 64, 64, 3, stride=1)

        # classifiers
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # temporal modules
        self.V1 = VTS(64, 32)
        self.V2 = VTS(128, 32)
        self.V3 = VTS(128, 32)

        # misc convolutions
        self.conv1 = conv1x1(128, 64)
        self.conv2 = conv1x1(128, 64)
        self.conv3 = conv1x1(128, 64)
        self.conv4 = conv1x1(128, 64)
        self.conv6 = conv1x1(128, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # weight initialization (as in original)
        initialize_weights(
            self.classifier1, self.classifier2, self.classifierCD,
            self.STC, self.V1, self.V2, self.conv1, self.conv2,
            self.F1, self.F2, self.conv3, self.conv4,
            self.conv6, self.SR, self.Up1, self.Up2, self.Up3, self.V3
        )

    @staticmethod
    def _make_layer(block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.GroupNorm(int(planes / 16), planes)
            )
        layers = [block(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def pair_to_video(self, x1: torch.Tensor, x2: torch.Tensor,
                      rate_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Interpolate between two frames to form a pseudo video."""
        def _interpolate(a, b, r_map, length):
            delta = 1.0 / (length - 1)
            delta_map = r_map * delta
            steps = torch.arange(length, dtype=torch.float, device=delta_map.device).view(1, -1, 1, 1, 1)
            return a.unsqueeze(1) + ((b - a) * delta_map).unsqueeze(1) * steps

        if rate_map is None:
            rate_map = torch.ones_like(x1[:, 0:1])
        return _interpolate(x1, x2, rate_map, self.video_len)

    @staticmethod
    def tem_aggr(f: torch.Tensor) -> torch.Tensor:
        """Aggregate temporal dimension with mean and max."""
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)

    def base_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from the backbone."""
        feats = []
        x = self.FCN.layer0(x)      # 64, 1/2
        x = self.FCN.maxpool(x)     # 64, 1/4
        feats.append(x)
        x = self.FCN.layer1(x)      # 64, 1/4
        x = self.FCN.layer2(x)      # 128, 1/8
        feats.append(x)
        x = self.FCN.layer3(x)      # 256, 1/8
        x = self.FCN.layer4(x)      # 512, 1/8
        x = self.FCN.head(x)        # 128, 1/8
        feats.append(x)
        return feats

    def CD_forward(self, D: torch.Tensor, T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Change detection decoder."""
        x = self.STC(D, T, S)
        u1 = self.Up1(x)
        u2 = u1 + S
        u2 = self.Up2(u2)
        u3 = u2 + S
        u3 = self.Up3(u3)
        change = self.classifierCD(u3)
        return change

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_size = x1.size()

        # multi-scale features
        f1 = self.base_forward(x1)
        f2 = self.base_forward(x2)

        # temporal features (VTS)
        T1 = self.pair_to_video(f1[0], f2[0]).permute(0, 2, 1, 3, 4)
        T1 = self.V1(T1)
        T1 = self.maxpool(self.conv1(self.tem_aggr(T1)))

        T2 = self.pair_to_video(f1[1], f2[1]).permute(0, 2, 1, 3, 4)
        T2 = self.V2(T2)
        T2 = self.conv2(self.tem_aggr(T2))

        T3 = self.pair_to_video(f1[2], f2[2]).permute(0, 2, 1, 3, 4)
        T3 = self.V3(T3)
        T3 = self.conv2(self.tem_aggr(T3))

        # spatial refinement (CotSR)
        d1, d2 = self.SR(f1[2], f2[2])
        D = self.conv4(torch.abs(d1 - d2))

        # multi-level difference features
        s1 = self.maxpool(torch.abs(f1[0] - f2[0]))
        s2 = self.conv3(torch.abs(f1[1] - f2[1]))
        s3 = self.conv3(torch.abs(f1[2] - f2[2]))

        # interaction with temporal features
        s1 = torch.sigmoid(s1 * T1) + s1
        T1 = torch.sigmoid(s1 * T1) + T1
        s2 = torch.sigmoid(s2 * T2) + s2
        T2 = torch.sigmoid(s2 * T2) + T2
        s3 = torch.sigmoid(s3 * T3) + s3
        T3 = torch.sigmoid(s3 * T3) + T3

        # fuse across levels
        S = self.F1(s1, s2, s3)
        T = self.F2(T1, T2, T3)

        change = self.CD_forward(T, D, S)

        out1 = self.classifier1(d1)
        out2 = self.classifier2(d2)

        change = F.upsample(change, x_size[2:], mode='bilinear')
        out1 = F.upsample(out1, x_size[2:], mode='bilinear')
        out2 = F.upsample(out2, x_size[2:], mode='bilinear')
        return change, out1, out2


# ------------------------------------------------------------------------------
# Quick test / FLOPs calculation
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    model = DFINet()
    x1 = torch.randn(1, 3, 512, 512)
    x2 = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, (x1, x2))
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
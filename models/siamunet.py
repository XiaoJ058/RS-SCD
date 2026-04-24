
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.old_model._blocks import Conv3x3, MaxPool2x2, ConvTransposed3x3


class KaimingInitMixin:
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # By default use fan_in mode and leaky relu non-linearity with a=0
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


Identity = nn.Identity


class SiamUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, use_dropout=False):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(3), nn.ReLU())
        self.use_dropout = use_dropout

        self.conv11 = Conv3x3(in_ch, 16, norm=True, act=True)
        self.do11 = self.make_dropout()
        self.conv12 = Conv3x3(16, 16, norm=True, act=True)
        self.do12 = self.make_dropout()
        self.pool1 = MaxPool2x2()

        self.conv21 = Conv3x3(16, 32, norm=True, act=True)
        self.do21 = self.make_dropout()
        self.conv22 = Conv3x3(32, 32, norm=True, act=True)
        self.do22 = self.make_dropout()
        self.pool2 = MaxPool2x2()

        self.conv31 = Conv3x3(32, 64, norm=True, act=True)
        self.do31 = self.make_dropout()
        self.conv32 = Conv3x3(64, 64, norm=True, act=True)
        self.do32 = self.make_dropout()
        self.conv33 = Conv3x3(64, 64, norm=True, act=True)
        self.do33 = self.make_dropout()
        self.pool3 = MaxPool2x2()

        self.conv41 = Conv3x3(64, 128, norm=True, act=True)
        self.do41 = self.make_dropout()
        self.conv42 = Conv3x3(128, 128, norm=True, act=True)
        self.do42 = self.make_dropout()
        self.conv43 = Conv3x3(128, 128, norm=True, act=True)
        self.do43 = self.make_dropout()
        self.pool4 = MaxPool2x2()

        self.upconv4 = ConvTransposed3x3(128, 128, output_padding=1)

        self.conv43d = Conv3x3(128, 128, norm=True, act=True)
        self.do43d = self.make_dropout()
        self.conv42d = Conv3x3(128, 128, norm=True, act=True)
        self.do42d = self.make_dropout()
        self.conv41d = Conv3x3(128, 64, norm=True, act=True)
        self.do41d = self.make_dropout()

        self.upconv3 = ConvTransposed3x3(64, 64, output_padding=1)

        self.conv33d = Conv3x3(64, 64, norm=True, act=True)
        self.do33d = self.make_dropout()
        self.conv32d = Conv3x3(64, 64, norm=True, act=True)
        self.do32d = self.make_dropout()
        self.conv31d = Conv3x3(64, 32, norm=True, act=True)
        self.do31d = self.make_dropout()

        self.upconv2 = ConvTransposed3x3(32, 32, output_padding=1)

        self.conv22d = Conv3x3(32, 32, norm=True, act=True)
        self.do22d = self.make_dropout()
        self.conv21d = Conv3x3(32, 16, norm=True, act=True)
        self.do21d = self.make_dropout()

        self.upconv1 = ConvTransposed3x3(16, 16, output_padding=1)

        self.conv12d = Conv3x3(16, 16, norm=True, act=True)
        self.do12d = self.make_dropout()
        self.conv11d = Conv3x3(16, out_ch)
        self.classifier1 = nn.Conv2d(128, 7, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, 7, kernel_size=1)

    def encode1(self, t):
        t_size = t.size()
        x11 = self.do11(self.conv11(t))
        x12_1 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12_1)

        # Stage 2
        x21 = self.do21(self.conv21(x1p))
        x22_1 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22_1)

        # Stage 3
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33_1 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33_1)

        # Stage 4
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43_1 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43_1)

        return x4p

    def encode2(self, t):
        t_size = t.size()
        x11 = self.do11(self.conv11(t))
        x12_1 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12_1)

        # Stage 2
        x21 = self.do21(self.conv21(x1p))
        x22_1 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22_1)

        # Stage 3
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33_1 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33_1)

        # Stage 4
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43_1 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43_1)

        return x4p

    def forward(self, t1, t2):
        # Encode
        # Stage 1
        t1_size = t1.size()
        t = torch.cat((t1, t2), dim=1)
        t = self.conv1(t)
        x4p = self.encode1(t)

        # Decode
        # Stage 4d
        x4d = self.upconv4(x4p)
        x43d = self.do43d(self.conv43d(x4d))
        x42d = self.do42d(self.conv42d(x43d))
        x41d = self.do41d(self.conv41d(x42d))

        # Stage 3d
        x3d = self.upconv3(x41d)
        x33d = self.do33d(self.conv33d(x3d))
        x32d = self.do32d(self.conv32d(x33d))
        x31d = self.do31d(self.conv31d(x32d))

        # Stage 2d
        x2d = self.upconv2(x31d)
        x22d = self.do22d(self.conv22d(x2d))
        x21d = self.do21d(self.conv21d(x22d))

        # Stage 1d
        x1d = self.upconv1(x21d)
        x12d = self.do12d(self.conv12d(x1d))
        x11d = self.conv11d(x12d)

        x43_1 = self.encode2(t1)
        x43_2 = self.encode2(t2)

        out1 = self.classifier1(x43_1)
        out2 = self.classifier2(x43_2)

        return x11d, F.upsample(out1, t1_size[2:], mode='bilinear'), F.upsample(out2, t1_size[2:], mode='bilinear')

    def make_dropout(self):
        if self.use_dropout:
            return nn.Dropout2d(p=0.2)
        else:
            return Identity()
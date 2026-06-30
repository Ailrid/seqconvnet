"""
Copyright (c) 2026-present Ailrid.
Licensed under the Apache License, Version 2.0.
Project: seqconvnet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(in_channels, num_groups=16):
    """GroupNorm模块,"""
    # 如果不能整除
    if in_channels % num_groups != 0:
        num_groups = in_channels
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.SiLU(),
            Normalize(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.SiLU(),
            Normalize(out_channels),
        )
        if self.in_channels != self.out_channels:
            # 如果输入输出通道不等，加一个卷积使残差可以连接
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        x0 = self.layer(x)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + x0


class DownSample(nn.Module):
    """平均池化下采样两倍或卷积缩小两倍"""

    def __init__(self, in_chs):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #    Transformer(in_chs, in_chs)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    """最临近插值二倍上采样或上采样后再进行一次卷积"""

    def __init__(self, in_channels, with_conv=False):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class AvgMaxAttention(nn.Module):

    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.avg = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            Normalize(out_chs),
        )
        self.max = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            Normalize(out_chs),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(2 * out_chs, in_chs, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x_avg = self.avg(x)
        x_max = self.max(x)
        score = self.mix(torch.cat((x_avg, x_max), dim=1))
        return x * F.sigmoid(score)


class ENetConv(nn.Module):
    """同UNet定义连续的俩次卷积"""

    def __init__(
        self,
        in_channels,
        out_channels,
        att=True,
    ):
        super(ENetConv, self).__init__()
        self.key = att
        self.conv = ResnetBlock(in_channels, out_channels)
        if att:
            self.att = AvgMaxAttention(out_channels, out_channels)
            # self.att = DotAttention(out_channels, out_channels)
            # self.att = Transformer(out_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.key:
            att = self.att(x)
            return x + att
        return x


class CustomConvEncoder(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
    ):
        features = [
            2 * in_channel,
            4 * in_channel,
            8 * in_channel,
            16 * in_channel,
            32 * in_channel,
        ]

        super(CustomConvEncoder, self).__init__()
        self.down1 = DownSample(features[0])
        self.down2 = DownSample(features[1])
        self.down3 = DownSample(features[2])
        self.down4 = DownSample(features[3])

        self.up1 = UpSample(features[4])
        self.up2 = UpSample(features[3])
        self.up3 = UpSample(features[2])
        self.up4 = UpSample(features[1])

        self.conv_down1 = ENetConv(in_channel, features[0])
        self.conv_down2 = ENetConv(features[0], features[1])
        self.conv_down3 = ENetConv(features[1], features[2])
        self.conv_down4 = ENetConv(features[2], features[3])
        self.conv_down5 = ENetConv(features[3], features[4])

        self.conv_up1 = ENetConv(features[3] + features[4], features[3])
        self.conv_up2 = ENetConv(features[2] + features[3], features[2])
        self.conv_up3 = ENetConv(features[1] + features[2], features[1])
        self.conv_up4 = ENetConv(features[0] + features[1], features[0])
        # self.final = nn.Sequential(
        #     nn.Conv2d(features[1], out_channel, kernel_size=3, stride=1, padding=1),
        #     nn.LayerNorm(out_channel),
        # )
        self.final = ENetConv(features[0], out_channel)

    def forward(self, x):
        x_down1 = self.conv_down1(x)
        x_down2 = self.conv_down2(self.down1(x_down1))
        x_down3 = self.conv_down3(self.down2(x_down2))
        x_down4 = self.conv_down4(self.down3(x_down3))
        x_down5 = self.conv_down5(self.down4(x_down4))

        x_up1 = self.conv_up1(torch.cat((x_down4, self.up1(x_down5)), dim=1))
        x_up2 = self.conv_up2(torch.cat((x_down3, self.up2(x_up1)), dim=1))
        x_up3 = self.conv_up3(torch.cat((x_down2, self.up3(x_up2)), dim=1))
        x_up4 = self.conv_up4(torch.cat((x_down1, self.up4(x_up3)), dim=1))

        return self.final(x_up4)

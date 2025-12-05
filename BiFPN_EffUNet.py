# contact : itstanmaypandey@gmail.com
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility blocks
# -----------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class MBConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expansion_rate=6, se=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_rate = expansion_rate
        self.se = se

        expansion_channels = in_channels * expansion_rate
        se_channels = max(1, int(in_channels * 0.25))

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            raise ValueError("unsupported kernel size")

        # Expansion
        if expansion_rate != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expansion_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.expand_conv = None

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=expansion_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=expansion_channels, bias=False),
            nn.BatchNorm2d(expansion_channels),
            nn.ReLU(inplace=True)
        )

        # Squeeze and excitation block
        if se:
            self.se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=expansion_channels, out_channels=se_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=se_channels, out_channels=expansion_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        else:
            self.se_block = None

        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=expansion_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):
        x = inputs
        if self.expand_conv is not None:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)

        if self.se_block is not None:
            x = self.se_block(x) * x

        x = self.pointwise_conv(x)

        if self.in_channels == self.out_channels and self.stride == 1:
            x = x + inputs

        return x


# -----------------------------
# BiFPN (4-level) - one layer
# -----------------------------
class BiFPNLayer(nn.Module):
    """
    4-level BiFPN layer operating on [P2, P3, P4, P5].
    All inputs are expected to have the same channel count.
    Implements fast normalized fusion with learnable positive weights.
    """
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.eps = eps
        C = channels

        # Top-down convs
        self.td_p4 = SeparableConv2d(C, C)
        self.td_p3 = SeparableConv2d(C, C)
        self.td_p2 = SeparableConv2d(C, C)

        # Bottom-up convs
        self.out_p3 = SeparableConv2d(C, C)
        self.out_p4 = SeparableConv2d(C, C)
        self.out_p5 = SeparableConv2d(C, C)

        # Learnable fusion weights (positivity enforced by ReLU then normalized)
        # Top-down nodes combine two inputs each (except P5 which is just itself)
        self.w_p4_td = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P4 + up(P5_td)
        self.w_p3_td = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P3 + up(P4_td)
        self.w_p2_td = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P2 + up(P3_td)

        # Bottom-up nodes
        self.w_p3_out = nn.Parameter(torch.ones(3, dtype=torch.float32))  # P3 + P3_td + down(P2_out)
        self.w_p4_out = nn.Parameter(torch.ones(3, dtype=torch.float32))  # P4 + P4_td + down(P3_out)
        self.w_p5_out = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P5 + down(P4_out)

    def _norm(self, w):
        w = F.relu(w)
        return w / (w.sum() + self.eps)

    def forward(self, p2, p3, p4, p5):
        # Top-down pathway
        # p5_td is just p5 (optionally could have a conv)
        p5_td = p5

        w = self._norm(self.w_p4_td)
        p4_td = w[0] * p4 + w[1] * F.interpolate(p5_td, scale_factor=2, mode='nearest')
        p4_td = self.td_p4(p4_td)

        w = self._norm(self.w_p3_td)
        p3_td = w[0] * p3 + w[1] * F.interpolate(p4_td, scale_factor=2, mode='nearest')
        p3_td = self.td_p3(p3_td)

        w = self._norm(self.w_p2_td)
        p2_td = w[0] * p2 + w[1] * F.interpolate(p3_td, scale_factor=2, mode='nearest')
        p2_td = self.td_p2(p2_td)

        # Bottom-up pathway
        p2_out = p2_td  # leaf

        w = self._norm(self.w_p3_out)
        p3_out = w[0] * p3 + w[1] * p3_td + w[2] * F.max_pool2d(p2_out, kernel_size=2)
        p3_out = self.out_p3(p3_out)

        w = self._norm(self.w_p4_out)
        p4_out = w[0] * p4 + w[1] * p4_td + w[2] * F.max_pool2d(p3_out, kernel_size=2)
        p4_out = self.out_p4(p4_out)

        w = self._norm(self.w_p5_out)
        p5_out = w[0] * p5 + w[1] * F.max_pool2d(p4_out, kernel_size=2)
        p5_out = self.out_p5(p5_out)

        return p2_out, p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    """Stack N BiFPN layers over 4 pyramid levels."""
    def __init__(self, channels, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([BiFPNLayer(channels) for _ in range(num_layers)])

    def forward(self, p2, p3, p4, p5):
        for layer in self.layers:
            p2, p3, p4, p5 = layer(p2, p3, p4, p5)
        return p2, p3, p4, p5


# -----------------------------
# EffUNet + BiFPN
# -----------------------------
class EffUNetBiFPN(nn.Module):
    """U-Net with EfficientNet-B0-like encoder + 4-level BiFPN decoder fusion.

    Args:
        in_channels (int): input channels
        classes (int): number of output classes (channels of final conv)
        fpn_channels (int): channel width for all pyramid levels inside BiFPN
        bifpn_layers (int): how many BiFPN layers to stack
    """
    def __init__(self, in_channels: int, classes: int, fpn_channels: int = 128, bifpn_layers: int = 2):
        super().__init__()
        self.fpn_channels = fpn_channels

        # ------- Encoder (your original) -------
        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.down_block_2 = nn.Sequential(
            MBConvBlock(32, 16, kernel_size=3, stride=1, expansion_rate=1),
            MBConvBlock(16, 24, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expansion_rate=6)
        )

        self.down_block_3 = nn.Sequential(
            MBConvBlock(24, 40, kernel_size=5, stride=2, expansion_rate=6),
            MBConvBlock(40, 40, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_4 = nn.Sequential(
            MBConvBlock(40, 80, kernel_size=3, stride=2, expansion_rate=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_rate=6),
            MBConvBlock(80, 112, kernel_size=5, stride=1, expansion_rate=6)
        )

        self.down_block_5 = nn.Sequential(
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(112, 192, kernel_size=5, stride=2, expansion_rate=6),  # -> 1/32
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_rate=6),
            MBConvBlock(192, 320, kernel_size=3, stride=1, expansion_rate=6)
        )

        # Encoder output channels at each scale
        self.c2_channels = 24   # x2 @ 1/4
        self.c3_channels = 40   # x3 @ 1/8
        self.c4_channels = 112  # x4 @ 1/16
        self.c5_channels = 320  # x5 @ 1/32
        self.c1_channels = 32   # x1 @ 1/2

        # ------- Lateral 1x1 to FPN width -------
        self.lateral_p2 = ConvBNAct(self.c2_channels, fpn_channels, k=1, s=1, p=0)
        self.lateral_p3 = ConvBNAct(self.c3_channels, fpn_channels, k=1, s=1, p=0)
        self.lateral_p4 = ConvBNAct(self.c4_channels, fpn_channels, k=1, s=1, p=0)
        self.lateral_p5 = ConvBNAct(self.c5_channels, fpn_channels, k=1, s=1, p=0)

        # ------- BiFPN stack -------
        self.bifpn = BiFPN(channels=fpn_channels, num_layers=bifpn_layers)

        # ------- Decoder on fused pyramid -------
        # We fuse as: y5=P5 -> up & concat P4 -> up & concat P3 -> up & concat P2 -> up & concat x1
        self.up_block_4 = DecoderBlock(in_channels=fpn_channels * 2, out_channels=256)   # up(y5)+P4
        self.up_block_3 =  DecoderBlock(in_channels=256 + fpn_channels, out_channels=128)  # up(y4)+P3
        self.up_block_2 =  DecoderBlock(in_channels=128 + fpn_channels, out_channels=64)   # up(y3)+P2
        self.up_block_1a = DecoderBlock(in_channels=64 + self.c1_channels, out_channels=32) # up(y2)+x1
        self.up_block_1b = DecoderBlock(in_channels=32, out_channels=16)

        self.head_conv = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=3, padding=1, bias=False)

    def _upsample(self, x, scale=2):
        return F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        x1 = self.start_conv(x)   # 1/2,  C1=32
        x2 = self.down_block_2(x1)  # 1/4, C2=24
        x3 = self.down_block_3(x2)  # 1/8, C3=40
        x4 = self.down_block_4(x3)  # 1/16, C4=112
        x5 = self.down_block_5(x4)  # 1/32, C5=320

        # Project to FPN channels
        p2 = self.lateral_p2(x2)
        p3 = self.lateral_p3(x3)
        p4 = self.lateral_p4(x4)
        p5 = self.lateral_p5(x5)

        # BiFPN fusion (stacked)
        p2, p3, p4, p5 = self.bifpn(p2, p3, p4, p5)

        # Decoder using fused pyramid
        y5 = p5
        y4 = self._upsample(y5)
        y4 = torch.cat([y4, p4], dim=1)
        y4 = self.up_block_4(y4)  # -> 1/16, 256ch

        y3 = self._upsample(y4)
        y3 = torch.cat([y3, p3], dim=1)
        y3 = self.up_block_3(y3)  # -> 1/8, 128ch

        y2 = self._upsample(y3)
        y2 = torch.cat([y2, p2], dim=1)
        y2 = self.up_block_2(y2)  # -> 1/4, 64ch

        y1 = self._upsample(y2)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.up_block_1a(y1)  # -> 1/2, 32ch

        y0 = self._upsample(y1)    # -> 1/1
        y0 = self.up_block_1b(y0)  # -> 16ch

        out = self.head_conv(y0)   # -> classes
        return out

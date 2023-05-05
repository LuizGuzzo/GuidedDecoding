import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pre_act=False, mish=False, ins=False):
        super(ConvBlock, self).__init__()
        self.pre_act = pre_act
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.mish = mish
        self.ins = ins
        self.padding = self.kernel // 2

        if not self.pre_act:
            if not self.mish and not self.ins:
                # conv2d -> ReLU
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.kernel, stride=1, padding=self.padding),
                    nn.ReLU(inplace=False)
                )
            elif self.mish and not self.ins:
                # conv2d -> Mish
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.kernel, stride=1, padding=self.padding),
                    nn.Mish()
                )
            elif not self.mish and self.ins:
                # conv2d -> IN -> ReLU
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.kernel, stride=1, padding=self.padding),
                    nn.InstanceNorm2d(self.out_ch),
                    nn.ReLU(inplace=False)
                )
            elif self.mish and self.ins:
                # conv2d -> IN -> Mish
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.kernel, stride=1, padding=self.padding),
                    nn.InstanceNorm2d(self.out_ch),
                    nn.Mish()
                )
        else:
            if not self.mish and not self.ins:
                # ReLU -> conv2d
                self.conv = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.kernel, stride=1, padding=self.padding)
                )
            elif self.mish and not self.ins:
                # Mish -> conv2d
                self.conv = nn.Sequential(
                    nn.Mish(),
                    nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.kernel, stride=1, padding=self.padding)
                )
            elif not self.mish and self.ins:
                # IN -> ReLU -> conv2d
                self.conv = nn.Sequential(
                    nn.InstanceNorm2d(self.in_ch),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.kernel, stride=1, padding=self.padding)
                )
            elif self.mish and self.ins:
                # IN -> Mish -> conv2d
                self.conv = nn.Sequential(
                    nn.InstanceNorm2d(self.in_ch),
                    nn.Mish(),
                    nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.kernel, stride=1, padding=self.padding)
                )

    def forward(self, x):
        out = self.conv(x)
        return out


def get_func(func_type, in_channel=16, out_channel=16):
    if func_type == 0:
        # 3x3 conv → ReLU
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=3, pre_act=False, mish=False, ins=False)
    elif func_type == 1:
        # 3x3 conv → Mish
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=3, pre_act=False, mish=True, ins=False)
    elif func_type == 2:
        # 3x3 conv → IN → ReLU
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=3, pre_act=False, mish=False, ins=True)
    elif func_type == 3:
        # 3x3 conv → IN → Mish
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=3, pre_act=False, mish=True, ins=True)
    elif func_type == 4:
        # 5x5 conv → ReLU
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=5, pre_act=False, mish=False, ins=False)
    elif func_type == 5:
        # 5x5 conv → Mish
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=5, pre_act=False, mish=True, ins=False)
    elif func_type == 6:
        # 5x5 conv → IN → ReLU
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=5, pre_act=False, mish=False, ins=True)
    elif func_type == 7:
        # 5x5 conv → IN → Mish
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=5, pre_act=False, mish=True, ins=True)
    elif func_type == 8:
        # ReLU → 3x3 conv
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=3, pre_act=True, mish=False, ins=False)
    elif func_type == 9:
        # Mish → 3x3 conv
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=3, pre_act=True, mish=True, ins=False)
    elif func_type == 10:
        # IN → ReLU → 3x3 conv
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=3, pre_act=True, mish=False, ins=True)
    elif func_type == 11:
        # IN → Mish → 3x3 conv
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=3, pre_act=True, mish=True, ins=True)
    elif func_type == 12:
        # ReLU → 5x5 conv
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=5, pre_act=True, mish=False, ins=False)
    elif func_type == 13:
        # Mish → 5x5 conv
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=5, pre_act=True, mish=True, ins=False)
    elif func_type == 14:
        # IN → ReLU → 5x5 conv
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=5, pre_act=True, mish=False, ins=True)
    elif func_type == 15:
        # IN → Mish → 5x5 conv
        func = ConvBlock(in_ch=in_channel, out_ch=out_channel, kernel=5, pre_act=True, mish=True, ins=True)
    else:
        raise ValueError(f"Invalid func_type: {func_type}")

    return func

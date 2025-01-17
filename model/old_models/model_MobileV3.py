from torch import nn


class HSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = self.relu(x + 3.) / 6.
        return out


class ConvBNAct(nn.Module):
    def __init__(self, inp, out, ks, act=nn.ReLU, bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(inp, out, ks, **kwargs)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.act = act(inplace=True) if act is not None else None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out) if self.bn is not None else out
        out = self.act(out) if self.act is not None else out

        return out


class SE(nn.Module):
    def __init__(self, inp, red=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.linear1 = nn.Linear(inp, inp // red, bias=False)
        self.nl1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(inp // red, inp, bias=False)
        self.nl2 = HSigmoid(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.pool(x).view(b, c)
        out = self.nl1(self.linear1(out))
        out = self.nl2(self.linear2(out)).view(b, c, 1, 1)
        return x * out.expand_as(x)

# SMALL_BBONE = BackboneConfig(
#     kernels=[3]*3 + [5]*8,
#    #in_channels = [16, 16, 24, 24, 40, 40, 40, 48, 48, 96, 96]
#     out_channels=[16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
#     expansion_sizes=[16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576],
#     ses=[True, False, False] + [True]*8,
#     nonlinearities=['relu6']*3 + ['hswish']*8,
#     strides=[2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1]
#     )
    # out_channels: List[int]
    # expansion_sizes: List[int]
    # strides: List[int]
    # kernels: List[int]
    # nonlinearities: List[str]
    # ses: List[bool]

# def __create_blocks(self, conf):
#         in_channels = [conf.out_channels[0]] + conf.out_channels[:-1]
#         return [Bneck(inp=in_channels[i],
#                       out=conf.out_channels[i],
#                       ks=conf.kernels[i],
#                       exp=conf.expansion_sizes[i],
#                       stride=conf.strides[i],
#                       nl=nls[conf.nonlinearities[i]],
#                       se=conf.ses[i])
#                 for i in range(len(conf.out_channels))]

class Bneck(nn.Module):
    def __init__(self, inp, out, ks, exp, stride, nl, se=False):
        assert stride in [1, 2], 'stride must be either 1 or 2'
        super().__init__()

        self.residual = stride == 1 and inp == out
        self.nl = nl(inplace=True)

        self.conv1 = ConvBNAct(inp, exp, 1, act=nl, bias=False)
        self.conv2 = ConvBNAct(exp, exp, ks, act=None, stride=stride,
                               padding=(ks - 1) // 2,
                               groups=exp, bias=False)
        self.conv3 = ConvBNAct(exp, out, 1, act=None, bias=False)
        self.se = SE(exp) if se else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        out = self.conv3(self.nl(out))

        if self.residual:
            out = out + x

        return 
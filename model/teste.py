# https://github.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch
from functools import partial
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm: nn.Module = nn.BatchNorm2d,
        act: nn.Module = nn.ReLU,
        **kwargs
    ):

        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            norm(out_features),
            act(),
        )


Conv1X1BnReLU = partial(ConvNormAct, kernel_size=1)
Conv3X3BnReLU = partial(ConvNormAct, kernel_size=3)


from torch import nn
from torch import Tensor

# class ResidualAdd(nn.Module):
#     def __init__(self, block: nn.Module):
#         super().__init__()
#         self.block = block
        
#     def forward(self, x: Tensor) -> Tensor:
#         res = x
#         x = self.block(x)
#         x += res
#         return x

    
from typing import Optional

class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x += res
        return x


from torch import nn
import math

class BottleNeck(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, reduction: int = 4):
        reduced_features = math.ceil(out_features / reduction)
        super().__init__(
            nn.Sequential(
                ResidualAdd(
                    nn.Sequential(
                        # wide -> narrow
                        Conv1X1BnReLU(in_features, reduced_features),
                        # narrow -> narrow
                        Conv3X3BnReLU(reduced_features, reduced_features),
                        # narrow -> wide
                        Conv1X1BnReLU(reduced_features, out_features, act=nn.Identity),
                    ),
                    shortcut=Conv1X1BnReLU(in_features, out_features)
                    if in_features != out_features
                    else None,
                ),
                nn.LeakyReLU(0.2), # troca para Relu
            )
        )


# class InvertedResidual(nn.Sequential):
#     def __init__(self, in_features: int, out_features: int, expansion: int = 4):
#         expanded_features = in_features * expansion
#         super().__init__(
#             nn.Sequential(
#                 ResidualAdd(
#                     nn.Sequential(
#                         # narrow -> wide
#                         Conv1X1BnReLU(in_features, expanded_features),
#                         # wide -> wide
#                         Conv3X3BnReLU(expanded_features, expanded_features),
#                         # wide -> narrow
#                         Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
#                     ),
#                     shortcut=Conv1X1BnReLU(in_features, out_features)
#                     if in_features != out_features
#                     else None,
#                 ),
#                 nn.ReLU(),
#             )
#         )
        
# class MobileNetLikeBlock(nn.Sequential):
#     def __init__(self, in_features: int, out_features: int, expansion: int = 4):
#         # use ResidualAdd if features match, otherwise a normal Sequential
#         residual = ResidualAdd if in_features == out_features else nn.Sequential
#         expanded_features = in_features * expansion
#         super().__init__(
#             nn.Sequential(
#                 residual(
#                     nn.Sequential(
#                         # narrow -> wide
#                         Conv1X1BnReLU(in_features, expanded_features),
#                         # wide -> wide
#                         Conv3X3BnReLU(expanded_features, expanded_features),
#                         # wide -> narrow
#                         Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
#                     ),
#                 ),
#                 nn.ReLU(),
#             )
#         )
        
# class DepthWiseSeparableConv(nn.Sequential):
#     def __init__(self, in_features: int, out_features: int):
#         super().__init__(
#             nn.Conv2d(in_features, in_features, kernel_size=3, groups=in_features),
#             nn.Conv2d(in_features, out_features, kernel_size=1)
#         )
        
# class MBConv(nn.Sequential):
#     def __init__(self, in_features: int, out_features: int, expansion: int = 4):
#         residual = ResidualAdd if in_features == out_features else nn.Sequential
#         expanded_features = in_features * expansion
#         super().__init__(
#             nn.Sequential(
#                 residual(
#                     nn.Sequential(
#                         # narrow -> wide
#                         Conv1X1BnReLU(in_features, 
#                                       expanded_features,
#                                       act=nn.ReLU6
#                                      ),
#                         # wide -> wide
#                         Conv3X3BnReLU(expanded_features, 
#                                       expanded_features, 
#                                       groups=expanded_features,
#                                       act=nn.ReLU6
#                                      ),
#                         # here you can apply SE
#                         # wide -> narrow
#                         Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
#                     ),
#                 ),
#                 nn.ReLU(),
#             )
#         )
        
class FusedMBConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, expansion: int = 4):
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        expanded_features = in_features * expansion
        super().__init__(
            nn.Sequential(
                residual(
                    nn.Sequential(
                        Conv3X3BnReLU(in_features, 
                                      expanded_features, 
                                      act=nn.ReLU6
                                     ),
                        # here you can apply SE
                        # wide -> narrow
                        Conv1X1BnReLU(expanded_features, out_features, act=nn.Identity),
                    ),
                ),
                nn.ReLU(),
            )
        )
        
# import torch
# import torch.nn as nn

# class PEM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(PEM, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

#     def forward(self, x, edge):
#         # Apply convolution to the input feature maps
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)

#         # Upsample the edge maps and concatenate with the feature maps
#         edge = nn.functional.interpolate(edge, size=x.shape[2:], mode='nearest')
#         x = torch.cat([x, edge], dim=1)

#         return x

# class EAM(nn.Module):
#     def __init__(self, in_channels):
#         super(EAM, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.conv_h = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

#     def forward(self, x, edge):
#         # Apply convolution to the input feature maps
#         x = self.conv1(x)

#         # Generate horizontal and vertical edge attention maps
#         edge_h = self.conv_h(edge)
#         edge_v = self.conv_v(edge)

#         # Apply attention maps to the feature maps
#         x_h = x * edge_h
#         x_v = x * edge_v

#         # Concatenate the attention maps and output
#         x = torch.cat([x_h, x_v], dim=1)
#         x = self.conv1(x)

#         return x

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

class BottleNeck(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, reduction: int = 4):
        reduced_features = out_features // reduction
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
                nn.ReLU(), # troca para leakyRelu
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
        

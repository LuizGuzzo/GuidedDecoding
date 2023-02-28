import torch
import torch.nn as nn
import torch.nn.functional as F
from .teste import *


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):        
        return self.convblock(x)

class ConvBlock_BottleNeck(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_BottleNeck, self).__init__()
        self.convblock = nn.Sequential(
            MobileNetLikeBlock(in_channels, out_channels),
            MobileNetLikeBlock(out_channels, out_channels)
            # nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):        
        return self.convblock(x)

# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         return out

class Up_concat(nn.Module):
    # upscale e convBlock

    def __init__(self, in_channels = None):
        super().__init__()

        # self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2) # sobe a resolução
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input, concat_with):
        up = self.up(input) 
        inter = F.interpolate(up, size=[concat_with[0].size(2), concat_with[0].size(3)], mode='bilinear', align_corners=True)
        concat_with.append(inter)
        concat = torch.cat(concat_with, dim=1)
        return concat
    
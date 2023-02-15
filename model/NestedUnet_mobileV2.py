import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

#codigo base: https://github.com/4uiiurz1/pytorch-nested-unet/blob/557ea02f0b5d45ec171aae2282d2cd21562a633e/archs.py

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):        
        return self.convblock(x)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

#                        bz, ch, hi, wi
# feature[0]: torch.Size([32, 3, 240, 320]) - 
# feature[1]: torch.Size([32, 32, 120, 160])
# feature[2]: torch.Size([32, 16, 120, 160])-
# feature[3]: torch.Size([32, 24, 60, 80])  
# feature[4]: torch.Size([32, 24, 60, 80])-  
# feature[5]: torch.Size([32, 32, 30, 40])  
# feature[6]: torch.Size([32, 32, 30, 40])  
# feature[7]: torch.Size([32, 32, 30, 40])-  
# feature[8]: torch.Size([32, 64, 15, 20])
# feature[9]: torch.Size([32, 64, 15, 20])
# feature[10]: torch.Size([32, 64, 15, 20])
# feature[11]: torch.Size([32, 64, 15, 20])
# feature[12]: torch.Size([32, 96, 15, 20])
# feature[13]: torch.Size([32, 96, 15, 20])
# feature[14]: torch.Size([32, 96, 15, 20])-
# feature[15]: torch.Size([32, 160, 8, 10])
# feature[16]: torch.Size([32, 160, 8, 10])
# feature[17]: torch.Size([32, 160, 8, 10])
# feature[18]: torch.Size([32, 320, 8, 10])
# feature[19]: torch.Size([32, 1280, 8, 10])- 

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        backbone_nn = models.mobilenet_v2( pretrained=True ) 
        
        print("NOT freezing backbone layers - MobileNetV2")
        for param in backbone_nn.parameters():
            param.requires_grad = True

        self.original_model = backbone_nn

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append( v(features[-1]) )
        return features

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

class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        # in_channels = [1280,96,32,24,16]
        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [3,16,24,32,96,1280] # troca para ser 320 em vez de 1280

        self.deep_supervision = deep_supervision

        
        self.upConcat = Up_concat()

        self.encoder = Encoder()


        self.conv0_1 = ConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv4_1 = ConvBlock(nb_filter[4]+nb_filter[5], nb_filter[4], nb_filter[4])

        self.conv0_2 = ConvBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_2 = ConvBlock(nb_filter[3]*2+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_3 = ConvBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_3 = ConvBlock(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_4 = ConvBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_4 = ConvBlock(nb_filter[1]*4+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_5 = ConvBlock(nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0])

        # self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final = ConvBlock(nb_filter[0], num_classes, num_classes)


    def forward(self, input):

        features = self.encoder(input)

        feats = [features[0],features[2],features[4],features[7],features[14],features[19]]

        x0_0 = feats[0]

        x1_0 = feats[1]
        x0_1 = self.conv0_1(self.upConcat(x1_0, [x0_0]))

        x2_0 = feats[2]
        x1_1 = self.conv1_1(self.upConcat(x2_0,[x1_0]))
        x0_2 = self.conv0_2(self.upConcat(x1_1,[x0_0, x0_1]))

        x3_0 = feats[3]
        x2_1 = self.conv2_1(self.upConcat(x3_0,[x2_0]))
        x1_2 = self.conv1_2(self.upConcat(x2_1,[x1_0, x1_1]))
        x0_3 = self.conv0_3(self.upConcat(x1_2,[x0_0, x0_1, x0_2]))

        x4_0 = feats[4]
        x3_1 = self.conv3_1(self.upConcat(x4_0,[x3_0]))
        x2_2 = self.conv2_2(self.upConcat(x3_1,[x2_0, x2_1]))
        x1_3 = self.conv1_3(self.upConcat(x2_2,[x1_0, x1_1, x1_2]))
        x0_4 = self.conv0_4(self.upConcat(x1_3,[x0_0, x0_1, x0_2, x0_3]))

        x5_0 = feats[5]
        x4_1 = self.conv4_1(self.upConcat(x5_0,[x4_0]))
        x3_2 = self.conv3_2(self.upConcat(x4_1,[x3_0, x3_1]))
        x2_3 = self.conv2_3(self.upConcat(x3_2,[x2_0, x2_1, x2_2]))
        x1_4 = self.conv1_4(self.upConcat(x2_3,[x1_0, x1_1, x1_2, x1_3]))
        x0_5 = self.conv0_5(self.upConcat(x1_4,[x0_0, x0_1, x0_2, x0_3, x0_4]))

        output = self.final(x0_5)
        return output
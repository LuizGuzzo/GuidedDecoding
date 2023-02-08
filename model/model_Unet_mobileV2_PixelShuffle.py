import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

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


def crop_img(source, target): # img menor , img maior (no upsampling)
    # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    diffX = target.size()[2] - source.size()[2]
    diffY = target.size()[3] - source.size()[3]

    # source = F.pad(source, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    # realizando o corte da imagem maior com o tamanho da img menor (proposto no paper do U-net)
    cropped_target = target[:,:,
                diffX//2:target.size()[2] - diffX//2,
                diffY//2:target.size()[3] - diffY//2
            ]
    return cropped_target


class Up(nn.Module):
    # upscale e convBlock

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.PixelShuffle(2)
        # dobro os canais porque sera processado a concatenação de 2 imagens
        self.conv = ConvBlock(in_channels*2, out_channels) 

    def forward(self, input, concat_with):

        inter = F.interpolate(input, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        concat = torch.cat([inter, concat_with], dim=1)
        x = self.conv(concat)
        return x

# class bridge(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         # self.bridge = nn.Sequential(
#         #     nn.MaxPool2d(2,2),
#         #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
#         #     nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         # )

#         self.max = nn.MaxPool2d(2,2)
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
#         self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
#     def forward(self, input):
#         # return self.bridge(input)
#         max = self.max(input)
#         conv = self.conv(max)
#         x = self.trans(conv)
#         return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        out_channels = [96,32,24,16,8]
        in_channels = [1280,96,32,24,16]
        
        # 16,24,40,80,112,160,960,1280
        # self.bridge = bridge(in_channels=960, out_channels=960) 
        self.bridge = nn.Conv2d(in_channels[0],in_channels[0], kernel_size=1, stride=1)
        # self.upList = []
        # for i in range(len(out_channels)):
        #     step = Up(in_channels=in_channels[i], out_channels=out_channels[i])
        #     self.upList.append(step)
        self.up0 = Up(in_channels=in_channels[0], out_channels=out_channels[0])
        self.up1 = Up(in_channels=in_channels[1], out_channels=out_channels[1])
        self.up2 = Up(in_channels=in_channels[2], out_channels=out_channels[2])
        self.up3 = Up(in_channels=in_channels[3], out_channels=out_channels[3])
        self.up4 = Up(in_channels=in_channels[4], out_channels=out_channels[4])

        self.conv = nn.ConvTranspose2d(out_channels[-1], 1, kernel_size=2, stride=2)

    def forward(self, features):

        # if True: # leitura de tamanho das features
        #     for block in range(len(features)):
        #         print("feature[{}]: {}".format(block,features[block].size()))

        # print("len Features:",len(features))
        
        #                        bz, ch, hi, wi
        # feature[0]: torch.Size([32, 3, 240, 320])
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

        # feats = [features[19],features[14],features[7],features[4],features[2]]

        x_d0 = self.bridge(features[19])

        # for i in range(len(self.upList)):
        #     stepConv = self.upList[i]
        #     x = stepConv(x,feats[i])

        x_d1 = self.up0(x_d0, features[19]) 
        x_d2 = self.up1(x_d1, features[14]) 
        x_d3 = self.up2(x_d2, features[7]) 
        x_d4 = self.up3(x_d3, features[4])
        x_d5 = self.up4(x_d4, features[2])

        x_d6 = self.conv(x_d5)
        
        return x_d6


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        backbone_nn = models.mobilenet_v2( pretrained=True ) 
        
        print("NOT freezing backbone layers - MobileNetV2")
        for param in backbone_nn.parameters():
            param.requires_grad = True

        # print(backbone_nn)
        # print("@@@ END BACKBONE @@@")

        #backbone._modules.classifier
        #backbone.classifier._modules
        self.original_model = backbone_nn

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append( v(features[-1]) )
        return features

class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.Unet = nn.Sequential(
            Encoder(),
            Decoder()
        )

    def forward(self, x):
        return self.Unet(x)

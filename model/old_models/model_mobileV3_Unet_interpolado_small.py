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


# processar 1280 > 1280

# upConv 1280 > 960
# processar 960 + 960 > 960

# upConv 960 > 160
# processar 160 + 160 > 160

# upConv 160 > 112

# conv_block (entrada, praXcanais)
#   processa a entrada convertendo para Xcanais #Conv2D

# decoder_block(entrada, skip_entrada, praXcanais)
#   aumenta a resolucao reduzindo os canais para Xcanais #Conv2DTranspose
#   concatena com a skip
#   vai para conv_block

class Up(nn.Module):
    # upscale e convBlock

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2) # sobe a resolução
        # dobro os canais porque sera processado a concatenação de 2 imagens
        self.conv = ConvBlock(in_channels*2, out_channels) 

    def forward(self, input, concat_with):
        # up = self.up(input) 
        # cropped = crop_img(concat_with,up) # invertido, errado
        # deveria estar expandindo o input (Width, hight), mas ja que o mobileNet nao expande a cada reducao de canais,
        # estou adaptando o input pro tamanho do concat
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

        out_channels = [48,24,16,16,8]
        in_channels = [576,48,24,16,16]
        
        # 16,24,40,80,112,160,960,1280
        # self.bridge = bridge(in_channels=960, out_channels=960) 
        self.bridge = nn.Conv2d(in_channels[0],in_channels[0], kernel_size=1, stride=1)
        self.up0 = Up(in_channels=in_channels[0], out_channels=out_channels[0])
        self.up1 = Up(in_channels=in_channels[1], out_channels=out_channels[1])
        self.up2 = Up(in_channels=in_channels[2], out_channels=out_channels[2])
        self.up3 = Up(in_channels=in_channels[3], out_channels=out_channels[3])
        self.up4 = Up(in_channels=in_channels[4], out_channels=out_channels[4])

        self.conv = nn.ConvTranspose2d(out_channels[-1], 1, kernel_size=2, stride=2)

    # def decoder_block(x,skip_input,num_filters):
    #     return nn.Sequential(
    #         # diminui os canais
    #         nn.conv_transpose2d(x, num_filters, kernel_size=3, stride=1, padding=1),
    #         # copia a resolução do skip_input (pega a res do mobilenet)
    #         F.interpolate(x, size=[skip_input.size(2), skip_input.size(3)], mode='bilinear', align_corners=True),
    #         # concatena os canais
    #         torch.cat([x, skip_input], dim=1),
    #         ConvBlock(x,num_filters)
    #     )

    def forward(self, features):

        # if True: # leitura de tamanho das features
        #     for block in range(len(features)):
        #         print("feature[{}]: {}".format(block,features[block].size()))

        # print("len Features:",len(features))
        
        #                        bz, ch, hi, wi
        # feature[0]: torch.Size([1, 3, 240, 320]) -
        # feature[1]: torch.Size([1, 16, 120, 160]) -
        # feature[2]: torch.Size([1, 16, 60, 80]) -
        # feature[3]: torch.Size([1, 24, 30, 40])
        # feature[4]: torch.Size([1, 24, 30, 40]) -
        # feature[5]: torch.Size([1, 40, 15, 20])
        # feature[6]: torch.Size([1, 40, 15, 20])
        # feature[7]: torch.Size([1, 40, 15, 20])
        # feature[8]: torch.Size([1, 48, 15, 20])
        # feature[9]: torch.Size([1, 48, 15, 20]) -
        # feature[10]: torch.Size([1, 96, 8, 10])
        # feature[11]: torch.Size([1, 96, 8, 10])
        # feature[12]: torch.Size([1, 96, 8, 10])
        # feature[13]: torch.Size([1, 576, 8, 10]) -      


        x_d0 = self.bridge(features[13])
        x_d1 = self.up0(x_d0, features[13]) 
        x_d2 = self.up1(x_d1, features[9]) 
        x_d3 = self.up2(x_d2, features[4]) 
        x_d4 = self.up3(x_d3, features[2])
        x_d5 = self.up4(x_d4, features[1])

        x_d6 = self.conv(x_d5)
        
        return x_d6


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        backbone_nn = models.mobilenet_v3_small( pretrained=True ) 
        
        print("NOT freezing backbone layers - MobileNetV3_Small")
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

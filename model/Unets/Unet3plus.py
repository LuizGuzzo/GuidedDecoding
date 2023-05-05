import torch.nn as nn
from .Encoder import Encoder

from .Blocks import *

from .teste import *


#                        bz, ch, he, wi
# feature[0]: torch.Size([16, 3, 240, 320])
# feature[1]: torch.Size([16, 32, 120, 160])
# feature[2]: torch.Size([16, 16, 120, 160])-
# feature[3]: torch.Size([16, 24, 60, 80])  
# feature[4]: torch.Size([16, 24, 60, 80])  -
# feature[5]: torch.Size([16, 32, 30, 40])  
# feature[6]: torch.Size([16, 32, 30, 40])  
# feature[7]: torch.Size([16, 32, 30, 40])  -
# feature[8]: torch.Size([16, 64, 15, 20])  
# feature[9]: torch.Size([16, 64, 15, 20])  
# feature[10]: torch.Size([16, 64, 15, 20]) 
# feature[11]: torch.Size([16, 64, 15, 20]) 
# feature[12]: torch.Size([16, 96, 15, 20]) 
# feature[13]: torch.Size([16, 96, 15, 20]) 
# feature[14]: torch.Size([16, 96, 15, 20]) -
# feature[15]: torch.Size([16, 160, 8, 10]) 
# feature[16]: torch.Size([16, 160, 8, 10])
# feature[17]: torch.Size([16, 160, 8, 10])
# feature[18]: torch.Size([16, 320, 8, 10])
# feature[19]: torch.Size([16, 1280, 8, 10])-

    

class UNet3plus(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=True , **kwargs):
        super().__init__()

        self.deep_supervision = deep_supervision
        
        nb_filter = [3,16,24,32,96,1280]
       
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder = Encoder()

        #Unet3+
        self.conv1 = FSA(nb_filter, nb_filter[0] )
        self.conv2 = FSA(nb_filter, nb_filter[1] )
        self.conv3 = FSA(nb_filter, nb_filter[2] )
        self.conv4 = FSA(nb_filter, nb_filter[3] )
        self.conv5 = FSA(nb_filter, nb_filter[4] )
        

        if self.deep_supervision:
            self.final1 = ConvBlock(nb_filter[0], num_classes)
            self.final2 = ConvBlock(nb_filter[1], num_classes)
            self.final3 = ConvBlock(nb_filter[2], num_classes)
            self.final4 = ConvBlock(nb_filter[3], num_classes)
            self.final5 = ConvBlock(nb_filter[4], num_classes)
            self.final6 = ConvBlock(nb_filter[5], num_classes)
        else:
            self.final = ConvBlock(nb_filter[0], num_classes)


    def forward(self, input):

        features = self.encoder(input)
        
        feats = [features[0],features[2],features[4],features[7],features[14],features[19]]

        x1e = feats[0]
        x2e = feats[1]
        x3e = feats[2]
        x4e = feats[3]
        x5e = feats[4]
        x6e = feats[5]

        x5d = self.conv5([x1e,x2e,x3e,x4e,x5e,x6e],x5e)
        x4d = self.conv4([x1e,x2e,x3e,x4e,x5d,x6e],x4e)
        x3d = self.conv3([x1e,x2e,x3e,x4d,x5d,x6e],x3e)
        x2d = self.conv2([x1e,x2e,x3d,x4d,x5d,x6e],x2e)
        x1d = self.conv1([x1e,x2d,x3e,x4d,x5d,x6e],x1e)

        if self.deep_supervision:
            output1 = self.final1(x1d)
            output2 = self.final2(x2d)
            output3 = self.final3(x3d)
            output4 = self.final4(x4d)
            output5 = self.final5(x5d)
            return [output1, output2, output3, output4, output5]

        else:
            output = self.final(x1d)
            return output

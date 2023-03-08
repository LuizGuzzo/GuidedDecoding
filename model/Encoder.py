import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        backbone_nn = models.mobilenet_v3_small( pretrained=True ) 
        
        print("NOT freezing backbone layers - MobileNetV3Small")
        for param in backbone_nn.parameters():
            param.requires_grad = True

        self.original_model = backbone_nn

    def forward(self, x):
        features = [x]
        for _, v in self.original_model.features._modules.items():
            features.append( v(features[-1]) )

        # if True: # leitura de tamanho das features
        #     for block in range(len(features)):
        #         print("feature[{}]: {}".format(block,features[block].size()))

        return features
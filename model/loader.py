import torch.nn as nn
import torch

# from model.Unet3plus import UNet3plus
from model.GAUnet.customUnet import GAUNet
from model.Unets.NestedUnet import NestedUNet

from model.old_models.GuideDepth import GuideDepth

def load_model(model_name, weights_pth,deep_supervision=False):
    model = model_builder(model_name,deep_supervision)

    if weights_pth is not None:
        print("Using pretrained weights at: ", weights_pth)
        state_dict = torch.load(weights_pth, map_location='cpu')
        model.load_state_dict(state_dict)

    return model

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000
    print("Parametros da rede: %.2f K" % count)

def model_builder(model_name,deep_supervision):
    # if model_name == 'GuideDepth':
    #     return GuideDepth(True)
    # if model_name == "teste": #'GuideDepth-S':
    #     modelo = GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])
    #     count_parameters(modelo)
    #     return modelo
    # if model_name == "teste":
    #     bin_list = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
    #     # print(len(bin_list))
    #     modelo = GAUNet(num_classes=1,input_channels=3,mid_channels=20,bin_genotype=bin_list).cuda()#.cpu()
    #     count_parameters(modelo)
    #     return modelo
    if model_name == "teste":
        modelo = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False).cuda()#.cpu()
        count_parameters(modelo)
        return modelo
    # if model_name == 'pixelformer':
    #     return PixelFormer(version="base07", inv_depth=True, max_depth=10, 
    #     pretrained="C:/Users/luizg/Documents/repositorios/GuidedDecoding/model/weights/swin_transformer/swin_base_patch4_window7_224_22k.pth")

    print("Invalid model")
    exit(0)



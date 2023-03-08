import torch.nn as nn
import torch

# from model.GuideDepth import GuideDepth
from model.NestedUnet_mobileV3Small import NestedUNet

def load_model(model_name, weights_pth):
    model = model_builder(model_name)

    if weights_pth is not None:
        print("Using pretrained weights at: ", weights_pth)
        state_dict = torch.load(weights_pth, map_location='cpu')
        model.load_state_dict(state_dict)

    return model

def model_builder(model_name):
    # if model_name == 'GuideDepth':
    #     return GuideDepth(True)
    # if model_name == 'GuideDepth-S':
    #     return GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])
    if model_name == "teste":
        return NestedUNet().cuda()#.cpu()
    # if model_name == 'pixelformer':
    #     return PixelFormer(version="base07", inv_depth=True, max_depth=10, 
    #     pretrained="C:/Users/luizg/Documents/repositorios/GuidedDecoding/model/weights/swin_transformer/swin_base_patch4_window7_224_22k.pth")

    print("Invalid model")
    exit(0)



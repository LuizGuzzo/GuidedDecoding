import torch.nn as nn
import torch

from model.GuideDepth import GuideDepth
from model.SAM.model_mobileV3_large_SAM import PTModel

def load_model(model_name, weights_pth):
    model = model_builder(model_name)

    if weights_pth is not None:
        state_dict = torch.load(weights_pth, map_location='cpu')
        model.load_state_dict(state_dict)

    return model

def model_builder(model_name):
    if model_name == 'GuideDepth':
        return GuideDepth(True)
    if model_name == 'GuideDepth-S':
        return GuideDepth(True, up_features=[32, 8, 4], inner_features=[32, 8, 4])
    if model_name == 'MobileNetV3':
        return PTModel()

    print("Invalid model")
    exit(0)



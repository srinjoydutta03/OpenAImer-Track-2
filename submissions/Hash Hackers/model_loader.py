import torch
from model import TinyResNet 

def load_model():
    model = TinyResNet(num_classes=100)
    model.load_state_dict(torch.load("model_final.pth", map_location="cpu"))
    model.eval()
    return model

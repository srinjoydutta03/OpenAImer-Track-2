import torch
from model import TinyResNet 

def load_model(model_path):
    model = TinyResNet(num_classes=100)
    model.load_state_dict(torch.load("model_path", map_location="cpu"))
    model.eval()
    return model

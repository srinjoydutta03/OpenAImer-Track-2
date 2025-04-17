# model_loader.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.quantization import quantize_dynamic

def load_model(model_path):
    model = resnet18()
    model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

import torch
from torchvision import models
import torch.nn as nn

def load_model(path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 100)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

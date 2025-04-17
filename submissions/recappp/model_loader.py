import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path):
    # Load standard resnet18 and remove layer4
    model = models.resnet18(pretrained=False)
    model.layer4 = nn.Identity()
    
    # Adjust fc layer to match layer3 output (256 channels)
    model.fc = nn.Linear(256, 100)

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model

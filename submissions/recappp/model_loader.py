import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path):
    # Load base ResNet18
    model = models.resnet18(pretrained=False)

    # Remove layer3 and layer4
    model.layer3 = nn.Identity()
    model.layer4 = nn.Identity()

    # Adjust fc to match layer2 output (128 channels)
    model.fc = nn.Linear(128, 100)

    # Load model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    return model

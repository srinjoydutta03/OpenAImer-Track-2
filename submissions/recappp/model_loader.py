import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path):
    # Load base ResNet18 without pre-trained weights
    model = models.resnet18(pretrained=False)

    # Remove layers 2, 3, and 4
    model.layer2 = nn.Identity()
    model.layer3 = nn.Identity()
    model.layer4 = nn.Identity()

    # Adjust final FC layer to match output of layer1
    model.fc = nn.Linear(model.layer1[-1].conv2.out_channels, 100)  # 100 classes

    # Load the saved model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    return model

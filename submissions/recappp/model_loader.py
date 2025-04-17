import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path, device='cpu'):
    # Load base ResNet18 without pre-trained weights
    model = models.resnet18(pretrained=False)

    # Keep only the first conv layer and remove all other layers
    model.conv1 = model.conv1  # 64 output channels
    model.bn1 = nn.Identity()
    model.relu = nn.Identity()
    model.maxpool = nn.Identity()
    model.layer1 = nn.Identity()
    model.layer2 = nn.Identity()
    model.layer3 = nn.Identity()
    model.layer4 = nn.Identity()

    # Replace avgpool and fc
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Optional: keep or remove
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 100)  # 100 classes
    )

    # Load the saved model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    return model



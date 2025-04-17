import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path):
    class CompressedResNet(nn.Module):
        def __init__(self, num_classes=100):
            super(CompressedResNet, self).__init__()
            base = models.resnet18(pretrained=False)
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = nn.Sequential(base.layer1[0])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    model = CompressedResNet(num_classes=100)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
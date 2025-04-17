import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ------------------------
# Model Definitions
# ------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_depthwise(self.depthwise(x)))
        x = self.bn_pointwise(self.pointwise(x))
        return self.relu(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)

class SmallerResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(3, 8, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(8, 16, stride=2)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.layer4 = self._make_layer(64, 64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_c, out_c, stride):
        return nn.Sequential(
            ResidualBlock(in_c, out_c, stride),
            ResidualBlock(out_c, out_c)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


# ------------------------
# Load Function
# ------------------------

def load_model(model_path):
    """
    Called by the evaluation script.
    Should return a torch.nn.Module (in eval mode) ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100
    model = SmallerResNet18(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model
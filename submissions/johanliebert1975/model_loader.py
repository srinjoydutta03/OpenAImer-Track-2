import torch
import torch.nn as nn

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
        return x

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

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class OptimizedResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_layer(64, 128, stride=2)
        self.conv3_x = self._make_layer(128, 256, stride=2)
        self.conv4_x = self._make_layer(256, 512, stride=2)
        self.conv5_x = self._make_layer(512, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

        # Add normalization values based on the training transform
        #self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        #self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _make_layer(self, in_c, out_c, stride):
        return nn.Sequential(
            ResidualBlock(in_c, out_c, stride),
            ResidualBlock(out_c, out_c)
        )

    def forward(self, x):
        # Move mean and std to the same device as the input tensor
        #mean = self.mean.to(x.device)
        #std = self.std.to(x.device)
        
        # Normalize input image (before the network processing)
        #x = (x - mean) / std
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def load_model(model_path):
    """
    Called by the evaluation script.
    Should return a torch.nn.Module (in eval mode) ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 100  # set this to your actual number of classes
    model = OptimizedResNet18(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model
import torch
     import torch.nn as nn
     import torchvision.models as models

     class SlimResNet18(nn.Module):
         def __init__(self, num_classes=100):
             super(SlimResNet18, self).__init__()
             base_model = models.resnet18(weights="IMAGENET1K_V1")
             self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
             self.bn1 = nn.BatchNorm2d(32)
             self.relu = nn.ReLU(inplace=True)
             self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
             self.layer1 = self._make_layer(base_model.layer1, in_channels=32, out_channels=32)
             self.layer2 = self._slim_layer(base_model.layer2, in_channels=32, out_channels=64)
             self.layer3 = self._slim_layer(base_model.layer3, in_channels=64, out_channels=128)
             self.layer4 = self._slim_layer(base_model.layer4, in_channels=128, out_channels=256)
             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
             self.fc = nn.Linear(256, num_classes)

         def _make_layer(self, layer, in_channels, out_channels):
             new_blocks = []
             for block in layer:
                 downsample = None
                 if block.downsample is not None or in_channels != out_channels:
                     downsample = nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=block.stride, bias=False),
                         nn.BatchNorm2d(out_channels)
                     )
                 new_block = models.resnet.BasicBlock(
                     inplanes=in_channels,
                     planes=out_channels,
                     stride=block.stride,
                     downsample=downsample
                 )
                 new_block.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=block.stride, padding=1, bias=False)
                 new_block.bn1 = nn.BatchNorm2d(out_channels)
                 new_block.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
                 new_block.bn2 = nn.BatchNorm2d(out_channels)
                 new_blocks.append(new_block)
                 in_channels = out_channels
             return nn.Sequential(*new_blocks)

         def _slim_layer(self, layer, in_channels, out_channels):
             new_blocks = []
             for block in layer:
                 downsample = None
                 if block.downsample is not None or in_channels != out_channels:
                     downsample = nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=block.stride, bias=False),
                         nn.BatchNorm2d(out_channels)
                     )
                 new_block = models.resnet.BasicBlock(
                     inplanes=in_channels,
                     planes=out_channels,
                     stride=block.stride,
                     downsample=downsample
                 )
                 new_block.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=block.stride, padding=1, bias=False)
                 new_block.bn1 = nn.BatchNorm2d(out_channels)
                 new_block.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
                 new_block.bn2 = nn.BatchNorm2d(out_channels)
                 new_blocks.append(new_block)
                 in_channels = out_channels
             return nn.Sequential(*new_blocks)

         def forward(self, x):
             x = self.conv1(x)
             x = self.bn1(x)
             x = self.relu(x)
             x = self.maxpool(x)
             x = self.layer1(x)
             x = self.layer2(x)
             x = self.layer3(x)
             x = self.layer4(x)
             x = self.avgpool(x)
             x = torch.flatten(x, 1)
             x = self.fc(x)
             return x
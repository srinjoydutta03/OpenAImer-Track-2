import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(MiniCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
		
def load_model(model_path="model.pth",device=None):
    if device==None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniCNN(num_classes=100)
    st = torch.load(model_path,weights_only=True,map_location=device)
    model.load_state_dict(st)
    model.to(device)
    model.eval()
    return model
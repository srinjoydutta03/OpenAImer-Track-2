import torch
import torch.nn as nn

class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / self.std
		

class MiniCNNv2(nn.Module):
    def __init__(self, num_classes=100):
        super(MiniCNNv2, self).__init__()
        self.normalize = NormalizeLayer(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
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
        x = self.normalize(x)   # 👈 Normalize inside model
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_model(model_path="model.pth",device=None):
    model = MiniCNNv2(100)
    if device==None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(model_path,weights_only=True,map_location=device)
    model.load_state_dict(st)
    model.to(device)
    model.eval()
    return model
import torch
import torch.nn as nn
from torchvision.models import resnet18

def load_model(model_path: str):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = resnet18()
    model.fc = nn.Linear(512, 100)
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print("Detected quantized model or mismatched keys, loading with relaxed constraints...")
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    model.eval()
    return model

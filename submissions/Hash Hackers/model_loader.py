import torch
import torch.nn as nn
import torchvision.models as models
import os

def load_model(checkpoint_path: str, num_classes: int = 100):
   
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle state_dict if present
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

if _name_ == "_main_":
    # Example usage
    model = load_model("model.pth")
    print("Model loaded successfully.")
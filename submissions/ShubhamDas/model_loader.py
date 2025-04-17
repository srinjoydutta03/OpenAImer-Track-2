# import os
import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import time

# Auto device selection
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"🖥️ Using device: {device}")

# Load compressed half-precision model
def load_model(model_path="model.pt",device=None):
    if device==None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

# Data transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# Dataset loader
# dataset = torchvision.datasets.ImageFolder(root='/content/archive/', transform=transform)
# loader = DataLoader(dataset, batch_size=32, shuffle=False)
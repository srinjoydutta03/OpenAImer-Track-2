import torch
from torchvision import transforms


def load_model(model_path):
  model = torch.jit.load(model_path, map_location='cpu')
  model.eval()
  return model

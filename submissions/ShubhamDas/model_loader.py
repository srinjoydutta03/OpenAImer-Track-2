import torch

def load_model(model_path="model.pt",device=None):
    if device==None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


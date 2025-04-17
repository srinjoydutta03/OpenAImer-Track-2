import torch

def load_model(model_path):
    """
    Custom model loader 
    
    Args:
        model_path (str): Path to the saved PyTorch model (.pth) file.

    Returns:
        model (torch.nn.Module): Loaded model, ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    return model

import torch
import time
#load model
def load_model(model_path,model_format=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    return model

# model_loader.py
import torch

def load_model(path='/kaggle/working/minimal_net_script.pt', device=None):
    if device is None:
        device = torch.device('cpu')
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model

if __name__ == '__main__':
    model = load_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)

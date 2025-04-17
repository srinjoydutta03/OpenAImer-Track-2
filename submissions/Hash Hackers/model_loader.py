# model_loader.py
import torch
import torchvision.models as models

def load_model(model_path, num_classes=100):
   
    
    model = models.resnet18(pretrained=False) 

    
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
   
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
        model.load_state_dict(state_dict) 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise

   
    model.eval()
    
    return model

if __name__ == "__main__":
    model_path = "model.pth"  
    model = load_model(model_path)

  
    input_tensor = torch.randn(1, 3, 224, 224)  

    with torch.no_grad(): 
        output = model(input_tensor)

    print("Model output:", output)

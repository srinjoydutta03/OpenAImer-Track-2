import torch
import torchvision.models as models
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path, num_classes=100, device='cpu', model_format=None):
    """
    Load a non-quantized ResNet18 model from model_path.
    Args:
        model_path (str): Path to model.pth
        num_classes (int): Number of output classes (default: 100)
        device (str): Device to load model on (default: 'cpu')
        model_format (str, optional): Model format (e.g., 'pytorch')
    Returns:
        torch.nn.Module: Loaded non-quantized model
    """
    try:
        logger.info(f"Loading non-quantized model from {model_path} on {device}")
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
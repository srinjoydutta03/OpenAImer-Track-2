import torch
import torch.nn as nn
import torchvision.models as models
import os

class ModelLoader:
    def _init_(self, architecture='resnet18', num_classes=100):
        """
        Initialize the model architecture.

        Args:
            architecture (str): Name of the architecture. Currently supports 'resnet18'.
            num_classes (int): Number of output classes.
        """
        self.architecture = architecture
        self.num_classes = num_classes
        self.model = self._initialize_model()

    def _initialize_model(self):
        """
        Initializes the model architecture.

        Returns:
            torch.nn.Module: The initialized model.
        """
        if self.architecture == 'resnet18':
            model = models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            raise NotImplementedError(f"Architecture '{self.architecture}' is not supported.")
        return model

    def load_weights(self, checkpoint_path):
        """
        Loads weights into the model.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        return self.model

    def get_model(self):
        """
        Returns the model instance.

        Returns:
            torch.nn.Module: The model.
        """
        return self.model

# Example usage
if _name_ == "_main_":
    loader = ModelLoader(architecture='resnet18', num_classes=100)
    model = loader.load_weights("model.pth")
    print("Model loaded successfully.")
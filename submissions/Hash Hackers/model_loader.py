import os
import torch
import torch.nn as nn
import torchvision.models as models


class ModelLoader:
    def _init_(
        self,
        architecture: str = "resnet18",
        num_classes: int = 100,
        device: str = None,
    ):
        """
        A flexible loader for torchvision-based models.

        Args:
            architecture (str): Model architecture, e.g. "resnet18".
            num_classes (int): Number of output logits.
            device (str): torch device string, e.g. "cuda:0" or "cpu".
        """
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._initialize_model().to(self.device)

    def _initialize_model(self) -> nn.Module:
        """Instantiate the chosen architecture and adjust its final layer."""
        if self.architecture == "resnet18":
            model = models.resnet18(pretrained=False)
            in_feats = model.fc.in_features
            model.fc = nn.Linear(in_feats, self.num_classes)
        else:
            raise NotImplementedError(
                f"Architecture '{self.architecture}' is not implemented."
            )
        return model

    def load_weights(self, checkpoint_path: str) -> nn.Module:
        """
        Load weights into the model, handling various checkpoint formats.

        Args:
            checkpoint_path (str): Path to the .pth checkpoint.

        Returns:
            nn.Module: The model with loaded weights, set to eval mode.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

        raw = torch.load(checkpoint_path, map_location="cpu")

        # Determine where the state_dict lives
        if isinstance(raw, dict):
            # common keys to look for
            for key in ("state_dict", "model_state_dict", "model"):
                if key in raw:
                    state_dict = raw[key]
                    break
            else:
                # assume the dict is the state_dict
                state_dict = raw
        elif isinstance(raw, torch.nn.Module):
            # checkpoint is a full model object
            self.model = raw.to(self.device)
            self.model.eval()
            return self.model
        else:
            raise RuntimeError(f"Unrecognized checkpoint format: {type(raw)}")

        # Clean any DataParallel "module." prefixes
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load into model
        self.model.load_state_dict(cleaned, strict=True)
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def get_model(self) -> nn.Module:
        """Return the model (already on correct device)."""
        return self.model


def load_model(
    checkpoint_path: str,
    architecture: str = "resnet18",
    num_classes: int = 100,
    device: str = None,
) -> torch.nn.Module:
    """
    Utility function to load a model in one call.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth file).
        architecture (str): Model architecture name, e.g. 'resnet18'.
        num_classes (int): Number of output classes.
        device (str, optional): Device specifier, e.g. 'cuda:0' or 'cpu'.

    Returns:
        torch.nn.Module: Loaded model in eval mode.
    """
    loader = ModelLoader(architecture=architecture, num_classes=num_classes, device=device)
    return loader.load_weights(checkpoint_path)
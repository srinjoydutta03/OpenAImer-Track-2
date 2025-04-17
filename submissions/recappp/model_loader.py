def load_model(model_path, device='cpu'):
    from torchvision import models
    import torch.nn as nn

    model = models.resnet18(pretrained=True)

    # Remove all residual layers
    model.layer1 = nn.Identity()
    model.layer2 = nn.Identity()
    model.layer3 = nn.Identity()
    model.layer4 = nn.Identity()

    # Keep stem as in training: conv1, bn1, relu, maxpool
    # Do NOT change bn1, relu, maxpool to Identity if you used them during training

    model.fc = nn.Linear(64, 100)  # Match the trained model's fc

    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    return model

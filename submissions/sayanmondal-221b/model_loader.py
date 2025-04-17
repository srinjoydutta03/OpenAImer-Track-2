import torch
import torch.nn as nn
import torch.quantization as tq
import torch.nn.utils.prune as prune
from torchvision.models import resnet18

def apply_structured_pruning(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')
    return model

def load_model(model_path):
    
    model = resnet18(pretrained=False)
    model = apply_structured_pruning(model, amount=0.5)

    
    model.qconfig = tq.get_default_qat_qconfig('fbgemm')
    tq.prepare_qat(model, inplace=True)

    tq.convert(model.eval(), inplace=True)

    
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    model.eval()
    return model

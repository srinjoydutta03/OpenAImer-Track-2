# ResNet18 Baseline Performance

This document provides the reference for the standard ResNet18 model, which serves as the baseline for the competition.

## Model Details

- **Architecture**: ResNet18
- **Parameters**: 11.7 million
- **Framework**: PyTorch (torchvision.models)
- **Input Size**: 224×224 RGB images
- **Output**: 100-class softmax probabilities


## Normalization Details

When submitting your compressed model, performance will be normalized relative to this baseline:

- **Size Score** = Baseline Size / Your Model Size
- **Latency Score** = Baseline Latency / Your Model Latency
- **Accuracy Score** = Your Model Accuracy / Baseline Accuracy

Higher normalized scores indicate better performance.

## Reference Implementation

The baseline model can be loaded with the following code:

```python
# PyTorch
import torch
import torchvision.models as models
model = models.resnet18(pretrained=True)
```
# TensorFlow
Tensorflow inherently doesn't support resnet18 models, you can find many implementations in github, kaggle, etc. You can use it to train your model and submit.

## Preprocessing Requirements

All models should use the same input preprocessing as the original ResNet18:

```python
# PyTorch preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# TensorFlow preprocessing
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image
``` 
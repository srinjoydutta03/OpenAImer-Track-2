# ResNet18 Baseline Performance

This document provides the reference performance metrics for the standard ResNet18 model, which serves as the baseline for the competition.

## Model Details

- **Architecture**: ResNet18
- **Parameters**: 11.7 million
- **Framework**: PyTorch (torchvision.models)
- **Input Size**: 224×224 RGB images
- **Output**: 1000-class softmax probabilities

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Model Size | 44.7 MB | Size of the serialized PyTorch model (.pt) |
| Inference Latency | 23.5 ms | Average over 1000 runs on evaluation hardware |
| Top-1 Accuracy | 69.76% | On ImageNet validation set |
| Top-5 Accuracy | 89.08% | On ImageNet validation set |

## Evaluation Hardware Specifications

- **CPU**: Intel Xeon E5-2680 v4 @ 2.40GHz
- **RAM**: 32GB
- **GPU**: NVIDIA Tesla T4 (used for evaluation)
- **Batch Size**: 1 (single image inference)

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

# TensorFlow
import tensorflow as tf
import tensorflow_hub as hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5")
])
```

## Preprocessing Requirements

All models should use the same input preprocessing as the original ResNet18:

```python
# PyTorch preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TensorFlow preprocessing
def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))
    image = tf.image.central_crop(image, 224/256)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return image
``` 
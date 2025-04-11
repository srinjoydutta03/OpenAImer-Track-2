# ResNet18 Model Compression Challenge

## Overview
Welcome to the ResNet18 Model Compression Challenge! This competition invites participants to compress a standard ResNet18 model while maintaining maximum performance. Your goal is to reduce model size and inference latency while preserving accuracy using techniques such as:

- Pruning
- Knowledge distillation
- Quantization
- Architecture optimization
- Any other compression method

## Competition Rules

### Model Requirements
- Base model: ResNet18
- Supported formats: PyTorch (.pt/.pth), TensorFlow (.h5/.pb/.saved_model), TFLite (.tflite)
- Model inputs/outputs must maintain the same interface as the original ResNet18

### Evaluation Criteria
Models will be evaluated on a weighted combination of:
1. **Model Size** (30%): The on-disk size of your model in MB
2. **Latency** (30%): Average inference time on our evaluation hardware
3. **Accuracy** (40%): Top-1 accuracy on the validation dataset

The final score is calculated as:
```
Final Score = (Size_Score × 0.3) + (Latency_Score × 0.3) + (Accuracy_Score × 0.4)
```

Each metric is normalized relative to the original ResNet18 baseline performance.

## Submission Process

### How to Submit
1. Fork this repository
2. Place your compressed model in the `submissions/` directory with the following structure:
   ```
   submissions/
   └── [your-github-username]/
       ├── model.{pt|h5|pb|saved_model|tflite}
       └── metadata.json
   ```
3. Create a `metadata.json` file with the following information:
   ```json
   {
     "username": "your-github-username",
     "model_format": "pytorch|tensorflow|tflite",
     "compression_technique": "Brief description of techniques used",
     "model_size_mb": 0.0,
     "additional_notes": "Optional: any special handling requirements"
   }
   ```
4. Submit a Pull Request (PR) to this repository with your submission

### Evaluation Process
- Our GitHub Actions workflow will automatically evaluate your model when you submit a PR
- The workflow downloads the validation dataset using secure credentials
- Models are evaluated for size, latency, and accuracy
- Results are automatically added to the leaderboard
- If you submit multiple PRs, only your best submission will be displayed

## Validation Dataset
The validation dataset is hosted privately on Kaggle and will be accessed securely during evaluation. The dataset consists of standard ImageNet validation images for classification.

## Leaderboard
The current leaderboard can be found in [LEADERBOARD.md](LEADERBOARD.md). It is automatically updated with each successful submission and ranks participants by their final score.

## Important Dates
- Competition Start: [Date]
- Submission Deadline: [Date]
- Winners Announcement: [Date]

## Technical Requirements
- Your model must load using standard PyTorch, TensorFlow, or TFLite loading functions
- Input preprocessing must match the original ResNet18 requirements
- Output format must match the original ResNet18 (1000-class softmax predictions)
- Models exceeding 500MB will be rejected

## Resources
- [Original ResNet18 Documentation](https://pytorch.org/hub/pytorch_vision_resnet/)
- [Model Compression Techniques Overview](https://arxiv.org/abs/1710.09282)
- [Baseline Performance Metrics](BASELINE.md)

## Questions and Support
For questions or support, please open an issue in this repository.

Good luck! 
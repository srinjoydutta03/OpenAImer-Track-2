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

Dataset will be resized to (224, 224) size only during evaluation. No other data augmentation has been applied, ensure your model supports it

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
       ├── model.{pt|pth|h5|pb|saved_model|tflite}
       └── metadata.json
       └── model_loader.py (optional)
       └── requirements.txt (optional)
   ```

   If you happen to create a complex model that doesn't support standard model loading as mentioned in the evaluate_script, you can provide a `model_loader.py` script of your own which will contain a `load_model` function that will load your model from your path. Also, any external dependencies that you might need specify them in `requirements.txt` file.
3. Create a `metadata.json` file with the following information:
   ```json
   {
     "username": "your-github-username",
     "teamname": "your-team-name",
     "model_format": "pytorch|tensorflow|tflite",
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
The validation dataset is hosted privately on Kaggle and will be accessed securely during evaluation. The dataset consists of standard validation images similar to the training set for classification.

## Leaderboard
The current leaderboard can be found in [LEADERBOARD.md](LEADERBOARD.md). It is automatically updated with each successful submission and ranks participants by their final score.

## Important Dates
- Competition Start: 12-April-2025 6:00 PM IST
- Submission Start: 17-April-2025 6:00 PM IST
- Submission Deadline: 17-April-2025 11:59 PM IST

#### PR'S SHOULD ONLY BE SUBMITTED IN THE PERIOD MENTIONED ABOVE, ANY PR'S BEFORE THE SUBMISSION START WILL BE DECLARED NULL AND VOID AND MAY INCUR NEGATIVE POINTS

## Technical Requirements
- Your model must load using standard PyTorch, TensorFlow, or TFLite loading functions
- Input preprocessing must match the original ResNet18 requirements
- Output format must match the original ResNet18 (100-class softmax predictions)

## Resources
- [Original ResNet18 Documentation](https://pytorch.org/hub/pytorch_vision_resnet/)
- [Model Compression Techniques Overview](https://arxiv.org/abs/1710.09282)
- [Baseline Performance Metrics](BASELINE.md)

## Questions and Support
For questions or support, please open an issue in this repository.

Good luck! 
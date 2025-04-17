# Compressed SlimResNet18 Model Submission

```
 This submission contains a compressed ResNet18 model (`SlimResNet18`) optimized for a 100-class classification task using structured pruning, knowledge distillation, and dynamic quantization.

 ## Files
 - `compressed_resnet18.pth`: PyTorch model weights (state dictionary).
 - `trained_pruned_architecture.json`: Architecture details after pruning and training.
 - `model_metadata.json`: Model model_metadata, including format and architecture file reference.
 - `model.py`: Definition of the `SlimResNet18` class.
 - `model_loader.py`: Custom loader script for model evaluation (loading, latency, accuracy, scoring).
 - `requirements.txt`: Required Python packages.
 - `README.md`: This documentation.

 ## Usage
 The evaluation panel will use `model_loader.py` to evaluate the model. To test manually:

 1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    The panel will automatically install these dependencies.

 2. **Run Evaluation**:
    ```bash
    python model_loader.py --submission_dir ./submission --data_dir /path/to/validation/data --baseline_size 50 --baseline_latency 10 --baseline_accuracy 80 --output_file results.json
    ```
    Replace `/path/to/validation/data` with the validation dataset path. Adjust baseline values as provided by the panel.

 3. **Model Details**:
    - **Architecture**: `SlimResNet18` (custom ResNet18 with reduced channels, e.g., conv1: 3->32).
    - **Compression**:
      - Structured pruning (30% channel reduction).
      - Knowledge distillation from ResNet50 (5 epochs, Adam optimizer, lr=0.001).
      - Dynamic quantization (qint8 for Conv2d and Linear layers).
    - **Input Shape**: [batch, 3, 224, 224] (RGB images, normalized with ImageNet mean/std).
    - **Output Shape**: [batch, 100] (100-class classification).
    - **Classes**: 100 classes.

 4. **Evaluation**:
    The `model_loader.py` script:
    - Loads the model using `compressed_resnet18.pth` and `metadata.json`.
    - Verifies the architecture with `trained_pruned_architecture.json`.
    - Measures model size (MB).
    - Evaluates latency (ms) over 100 runs.
    - Computes accuracy (%) on the validation dataset.
    - Calculates scores (size, latency, accuracy, total) relative to baseline values.
    - Saves results to a JSON file (e.g., `results.json`).

 5. **Architecture File**:
    The `trained_pruned_architecture.json` file details:
    - Model type (`SlimResNet18`) and number of classes (100).
    - Pruning rate (0.3) and compression steps.
    - Input shape ([3, 224, 224]) and output shape ([100]).
    - Training details (distillation from ResNet50, 5 epochs).
    - Layer configurations (e.g., in_channels, out_channels, kernel_size) and pruning information.

 ## Notes
 - The model expects inputs preprocessed with:
   - Resize to 256x256, center crop to 224x224.
   - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
 - The `model_loader.py` script replicates the panel’s evaluation logic, ensuring compatibility with the `SlimResNet18` model.
 - All file paths are relative to the submission directory for portability.
 - The panel’s loader will call the `load_model` function in `model_loader.py` for model loading, but the script can also run standalone for full evaluation.

 ## Contributor
 Shubham Das
```
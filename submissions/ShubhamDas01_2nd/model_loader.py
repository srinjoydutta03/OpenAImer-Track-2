import os
import json
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from model import SlimResNet18  # Assumes SlimResNet18 is defined in model.py

def get_model_size_mb(model_path):
    """Get the size of the model file in MB."""
    return os.path.getsize(model_path) / (1024 * 1024)

def load_model(model_path):
    """
    Load the compressed SlimResNet18 model using the weights and metadata.
    
    Args:
        model_path (str): Path to the model weights file (.pth).
    
    Returns:
        torch.nn.Module: Loaded PyTorch model ready for inference.
    """
    submission_dir = os.path.dirname(model_path)
    
    # Load metadata
    metadata_path = os.path.join(submission_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    try:
        with open(metadata_path, "r") as f:
            model_metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load metadata: {e}")
    
    # Verify model format
    model_format = model_metadata.get("model_format", "pytorch")
    if model_format not in ["pytorch", "pt", "pth"]:
        raise ValueError(f"Unsupported model format: {model_format}")
    
    # Get number of classes
    num_classes = model_metadata.get("num_classes", 100)
    
    # Initialize model
    try:
        model = SlimResNet18(num_classes=num_classes)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize SlimResNet18: {e}")
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint  # In case the full model was saved
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    model.eval()
    
    # Verify architecture
    architecture_path = model_metadata.get("architecture_file", os.path.join(submission_dir, "trained_pruned_architecture.json"))
    if os.path.exists(architecture_path):
        try:
            with open(architecture_path, "r") as f:
                architecture = json.load(f)
            print(f"Loaded architecture: {architecture['model_type']} with {architecture['num_classes']} classes")
            print(f"Pruning rate: {architecture['pruning_rate']}")
            print(f"Number of layers: {len(architecture['layers'])}")
        except Exception as e:
            print(f"Warning: Failed to load architecture file: {e}")
    
    return model

def prepare_data_loader(data_dir, batch_size=1):
    """Prepare data loader for the validation dataset."""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # Force 3 channels
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    
    dataset = ImageFolder(root=data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def measure_latency(model, data_loader, num_runs=100):
    """Measure inference latency of the model."""
    print("Measuring latency ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Warm up
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)
        break
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_runs:
                break
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            latencies.append((time.time() - start_time) * 1000)  # Convert to ms
    
    return np.mean(latencies)

def evaluate_accuracy(model, data_loader):
    """Evaluate model accuracy on the validation set."""
    print("Evaluating accuracy ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def calculate_scores(model_size, latency, accuracy, baseline_size, baseline_latency, baseline_accuracy):
    """Calculate normalized scores for each metric."""
    size_score = baseline_size / model_size  # Smaller is better
    latency_score = baseline_latency / latency  # Faster is better
    accuracy_score = accuracy / baseline_accuracy  # Higher is better
    
    # Weighted total score (30% size, 30% latency, 40% accuracy)
    total_score = 0.3 * size_score + 0.3 * latency_score + 0.4 * accuracy_score
    
    return {
        'size_score': size_score,
        'latency_score': latency_score,
        'accuracy_score': accuracy_score,
        'total_score': total_score
    }

def evaluate_submission(model_path, data_dir, baseline_size, baseline_latency, baseline_accuracy, output_file):
    """
    Evaluate the model submission, performing all tasks of the panel's loader.
    
    Args:
        model_path (str): Path to the model weights file (.pth).
        data_dir (str): Directory containing validation data.
        baseline_size (float): Baseline model size in MB.
        baseline_latency (float): Baseline model latency in ms.
        baseline_accuracy (float): Baseline model accuracy in %.
        output_file (str): Path to save evaluation results (JSON).
    
    Returns:
        dict: Evaluation results (size, latency, accuracy, scores).
    """
    # Get model size
    model_size = get_model_size_mb(model_path)
    
    # Load model
    model = load_model(model_path)
    
    # Prepare data loader
    data_loader = prepare_data_loader(data_dir)
    
    # Measure latency
    latency = measure_latency(model, data_loader)
    
    # Evaluate accuracy
    accuracy = evaluate_accuracy(model, data_loader)
    
    # Calculate scores
    scores = calculate_scores(
        model_size, latency, accuracy,
        baseline_size, baseline_latency, baseline_accuracy
    )
    
    # Combine results
    results = {
        'model_size': model_size,
        'latency': latency,
        'accuracy': accuracy,
        'size_score': scores['size_score'],
        'latency_score': scores['latency_score'],
        'accuracy_score': scores['accuracy_score'],
        'total_score': scores['total_score']
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    print(f"Evaluation completed. Results saved to {output_file}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate SlimResNet18 model submission')
    parser.add_argument('--submission_dir', type=str, required=True, help='Directory containing the submission')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing validation data')
    parser.add_argument('--baseline_size', type=float, required=True, help='Baseline model size in MB')
    parser.add_argument('--baseline_latency', type=float, required=True, help='Baseline model latency in ms')
    parser.add_argument('--baseline_accuracy', type=float, required=True, help='Baseline model accuracy in %')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file for evaluation results')
    
    args = parser.parse_args()
    
    # Construct model path
    model_path = os.path.join(args.submission_dir, "compressed_resnet18.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    evaluate_submission(
        model_path=model_path,
        data_dir=args.data_dir,
        baseline_size=args.baseline_size,
        baseline_latency=args.baseline_latency,
        baseline_accuracy=args.baseline_accuracy,
        output_file=args.output_file
    )
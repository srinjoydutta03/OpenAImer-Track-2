#!/usr/bin/env python3
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
import json

# Import conditionally based on model format
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


def get_model_size_mb(model_path):
    """Get the size of the model file in MB."""
    return os.path.getsize(model_path) / (1024 * 1024)


def load_model(model_path, model_format):
    """Load the model based on its format."""
    submission_dir = os.path.dirname(model_path)
    
    # Check for custom requirements.txt and install dependencies
    requirements_path = os.path.join(submission_dir, 'requirements.txt')
    if os.path.exists(requirements_path):
        print(f"Installing custom requirements from {requirements_path}")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            print("Successfully installed custom requirements")
        except Exception as e:
            print(f"Warning: Failed to install custom requirements: {e}")
    
    # Check for custom model loader - works for any model format
    loader_path = os.path.join(submission_dir, 'model_loader.py')
    if os.path.exists(loader_path):
        try:
            import sys
            import importlib.util
            
            # Add submission directory to path
            if submission_dir not in sys.path:
                sys.path.insert(0, submission_dir)
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("model_loader", loader_path)
            loader_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader_module)
            
            if hasattr(loader_module, 'load_model'):
                print(f"Using custom model loader from {loader_path}")
                model = loader_module.load_model(model_path)
                return model
            else:
                print(f"Warning: model_loader.py exists but doesn't contain a load_model function")
        except Exception as e:
            print(f"Error using custom model loader: {e}")
            print("Falling back to standard model loading")
    if model_format in ['pytorch', 'pt', 'pth']:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to load this model format")
        
        # Load the file
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if it's a state dict or full model
        if isinstance(checkpoint, dict) and not hasattr(checkpoint, 'to'):
            # This is a state dict, need to create the model first
            import torchvision.models as models
            
            # Read metadata to get architecture if available
            metadata_path = os.path.join(os.path.dirname(model_path), 'metadata.json')
            
            architecture = 'resnet18'  # Default
            num_classes = 100  # Default
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Try to infer num_classes from the checkpoint if not in metadata
            if 'fc.weight' in checkpoint:
                inferred_classes = checkpoint['fc.weight'].size(0)
                num_classes = inferred_classes
                print(f"Detected {num_classes} output classes from model weights")
            elif 'state_dict' in checkpoint and 'fc.weight' in checkpoint['state_dict']:
                inferred_classes = checkpoint['state_dict']['fc.weight'].size(0)
                num_classes = inferred_classes
                print(f"Detected {num_classes} output classes from model weights")
            
            # Initialize the model based on architecture with the correct number of classes
            if hasattr(models, architecture):
                model = getattr(models, architecture)(num_classes=num_classes)
                
                # Handle both direct state dict and checkpoint with state_dict key
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                return model
            else:
                raise ValueError(f"Unsupported model architecture: {architecture}")
        else:
            return checkpoint
    
    elif model_format in ['tensorflow', 'tf', 'h5', 'pb', 'saved_model']:
        if not HAS_TF:
            raise ImportError("TensorFlow is required to load this model format")
        # Load the model with custom objects
        return tf.keras.models.load_model(model_path)
    
    elif model_format in ['tflite']:
        if not HAS_TF:
            raise ImportError("TensorFlow is required to load this model format")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    else:
        raise ValueError(f"Unsupported model format: {model_format}")


def measure_latency(model, model_format, data_loader, num_runs=100):
    """Measure inference latency of the model."""
    # Get a sample input
    if model_format in ['pytorch', 'pt', 'pth']:
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
                latencies.append((time.time() - start_time) * 1000)  # convert to ms
    
    elif model_format in ['tensorflow', 'tf', 'h5', 'pb', 'saved_model']:
        # Warm up
        for images, _ in data_loader:
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            _ = model.predict(images)
            break
        
        # Measure latency
        latencies = []
        for i, (images, _) in enumerate(data_loader):
            if i >= num_runs:
                break
            
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            
            start_time = time.time()
            _ = model.predict(images)
            latencies.append((time.time() - start_time) * 1000)  # convert to ms
    
    elif model_format in ['tflite']:
        interpreter = model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warm up
        for images, _ in data_loader:
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            interpreter.set_tensor(input_details[0]['index'], images)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            break
        
        # Measure latency
        latencies = []
        for i, (images, _) in enumerate(data_loader):
            if i >= num_runs:
                break
            
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], images)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            latencies.append((time.time() - start_time) * 1000)  # convert to ms
    
    # Return average latency
    return np.mean(latencies)


def evaluate_accuracy(model, model_format, data_loader):
    """Evaluate model accuracy on the validation set."""
    correct = 0
    total = 0
    
    if model_format in ['pytorch', 'pt', 'pth']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
    elif model_format in ['tensorflow', 'tf', 'h5', 'pb', 'saved_model']:
        for images, labels in data_loader:
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            
            predictions = model.predict(images)
            predicted_classes = np.argmax(predictions, axis=1)
            total += labels.shape[0]
            correct += np.sum(predicted_classes == labels)
    
    elif model_format in ['tflite']:
        interpreter = model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        for images, labels in data_loader:
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            
            interpreter.set_tensor(input_details[0]['index'], images)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_classes = np.argmax(predictions, axis=1)
            total += labels.shape[0]
            correct += np.sum(predicted_classes == labels)
    
    accuracy = 100 * correct / total
    return accuracy


def prepare_data_loader(data_dir, model_format, batch_size=1):
    """Prepare data loader for the validation dataset."""
    if model_format in ['pytorch', 'pt', 'pth']:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        dataset = ImageFolder(root=data_dir, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    elif model_format in ['tensorflow', 'tf', 'h5', 'pb', 'saved_model', 'tflite']:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        dataset = ImageFolder(root=data_dir, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader


def calculate_scores(model_size, latency, accuracy, baseline_size, baseline_latency, baseline_accuracy):
    """Calculate normalized scores for each metric."""
    size_score = baseline_size / model_size  # Smaller is better
    latency_score = baseline_latency / latency  # Faster is better
    accuracy_score = accuracy / baseline_accuracy  # Higher is better
    
    # Calculate weighted total score (30% size, 30% latency, 40% accuracy)
    total_score = 0.3 * size_score + 0.3 * latency_score + 0.4 * accuracy_score
    
    return {
        'size_score': size_score,
        'latency_score': latency_score,
        'accuracy_score': accuracy_score,
        'total_score': total_score
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model submission')
    parser.add_argument('--submission_dir', type=str, required=True, help='Directory containing the submission')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing validation data')
    parser.add_argument('--baseline_size', type=float, required=True, help='Baseline model size in MB')
    parser.add_argument('--baseline_latency', type=float, required=True, help='Baseline model latency in ms')
    parser.add_argument('--baseline_accuracy', type=float, required=True, help='Baseline model accuracy in %')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file for evaluation results')
    
    args = parser.parse_args()
    
    # Load metadata
    with open(os.path.join(args.submission_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    model_format = metadata['model_format']
    
    # Find model file
    model_files = list(Path(args.submission_dir).glob('model.*'))
    if not model_files:
        raise FileNotFoundError("Model file not found in submission directory")
    
    model_path = str(model_files[0])
    
    # Get model size
    model_size = get_model_size_mb(model_path)
    
    # Load model
    model = load_model(model_path, model_format)
    
    # Prepare data loader
    data_loader = prepare_data_loader(args.data_dir, model_format)
    
    # Measure latency
    latency = measure_latency(model, model_format, data_loader)
    
    # Evaluate accuracy
    accuracy = evaluate_accuracy(model, model_format, data_loader)
    
    # Calculate scores
    scores = calculate_scores(
        model_size, latency, accuracy,
        args.baseline_size, args.baseline_latency, args.baseline_accuracy
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
    with open(args.output_file, 'w') as f:
        json.dump(results, f)
    
    print(f"Evaluation completed. Results saved to {args.output_file}")


if __name__ == '__main__':
    main() 
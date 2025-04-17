import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# Auto device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Load compressed half-precision model
model = torch.jit.load('/content/final_resnet18_compressed.pt', map_location=device)
model.eval()

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset loader
dataset = torchvision.datasets.ImageFolder(root='/content/archive/', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Accuracy calculation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device).half()  # Convert input to half precision
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

# Latency calculation
input_sample = torch.randn(1, 3, 224, 224).to(device).half()  # Half precision input

# Warm-up
for _ in range(10):
    _ = model(input_sample)

# Measure
start_time = time.time()
for _ in range(100):
    _ = model(input_sample)
latency_ms = (time.time() - start_time) / 100 * 1000

# Model Size
model_path = '/content/final_resnet18_compressed.pt'
model_size_mb = os.path.getsize(model_path) / 1e6

# Print all metrics together
print("\n📊 Final Compressed Model Evaluation:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average Latency: {latency_ms:.2f} ms")
print(f"Model Size: {model_size_mb:.2f} MB")
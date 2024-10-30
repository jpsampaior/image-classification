import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn


# Prepare to use ResNet-18 (Resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
val_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


def filter_by_class_limit(dataset, class_limit):
    filtered_data = []
    # Dictionary to count the number of images in each class
    class_counts = defaultdict(int)

    for img, label in dataset:
        # If the limit wasn't achieved, add
        if class_counts[label] < class_limit:
            filtered_data.append((img, label))
            class_counts[label] += 1
        # If we achieved our goal in all classes, break the loop
        if all(count >= class_limit for count in class_counts.values()):
            break

    return filtered_data


# Filter images to get the first 500 training images and 100 test images per class
filtered_train_data = filter_by_class_limit(train_data, class_limit=10)
filtered_val_data = filter_by_class_limit(val_data, class_limit=5)

# Checking if the images were loaded as needed
print("Number of training images:", len(filtered_train_data))
print("Number of test images:", len(filtered_val_data))

# Creating DataLoaders - Question: Why do that?
train_loader = DataLoader(filtered_train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(filtered_val_data, batch_size=32, shuffle=False)

# Load ResNet-18 and remove the last layer
model = resnet18()
model.fc = nn.Identity()  # Replace the last layer with an identity layer
model.eval()  # Question: Why do that?

# Question: Why do that?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Feature extraction function
def extract_features(data_loader):
    features = []
    # No Gradient to speed the process
    with torch.no_grad():
        for imgs, lbls in data_loader:
            # Extract features using ResNet18 model
            features.append(model(imgs))
    return torch.cat(features)


# Extracting features
train_features = extract_features(train_loader)
val_features = extract_features(val_loader)

# Verify features extraction
print("Training features shape:", train_features.shape)
print("Validation features shape:", val_features.shape)


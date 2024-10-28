import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict

tensor_transform = transforms.Compose([transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tensor_transform)
val_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tensor_transform)

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


# Filter images
filtered_train_data = filter_by_class_limit(train_data, class_limit=500)
filtered_val_data = filter_by_class_limit(val_data, class_limit=100)

print("Number of training images:", len(filtered_train_data))
print("Number of test images:", len(filtered_val_data))

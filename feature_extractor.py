import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm


def filter_by_class_limit(dataset, class_limit):
    filtered_data = []
    class_counts = defaultdict(int)
    for img, label in dataset:
        if class_counts[label] < class_limit:
            filtered_data.append((img, label))
            class_counts[label] += 1
        if all(count >= class_limit for count in class_counts.values()):
            break
    return filtered_data


class FeatureExtractor:
    def __init__(self, train_class_limit=500, test_class_limit=100, batch_size=32, pca_components=50):
        self.train_class_limit = train_class_limit
        self.test_class_limit = test_class_limit
        self.batch_size = batch_size
        self.pca_components = pca_components

        # Prepare to resize images to 224x224x3
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load dataset (CIFAR10) using the resize option
        self.train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                                       transform=self.transform)
        self.test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                                     transform=self.transform)

        # Filter the data to get 500 training images (50 of each class) and 100 test images (10 of each class)
        print(f"\nFiltering data to get {train_class_limit*10} training images and {test_class_limit*10} testing images...")
        self.filtered_train_data = filter_by_class_limit(self.train_data, self.train_class_limit)
        self.filtered_test_data = filter_by_class_limit(self.test_data, self.test_class_limit)
        print("Number of training images:", len(self.filtered_train_data))
        print("Number of test images:", len(self.filtered_test_data))

        # Creating dataloaders
        self.train_loader = DataLoader(self.filtered_train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.filtered_test_data, batch_size=self.batch_size, shuffle=False)

        # Initializing the Resnet18 Model to extract the features vectors
        self.model = resnet18()
        self.model.fc = nn.Identity()  # Remove the last layer of Resnet18
        self.model.eval()

    # Function used to extract the features
    def extract_features(self, data_loader):
        features = []
        with torch.no_grad():
            for imgs, _ in tqdm(data_loader, desc="Extracting Features"):
                features.append(self.model(imgs))
        return torch.cat(features)  # Concatenating the multiple batches

    # Function used to apply the PCA to reduce the size of feature vectors from 512×1 50×1
    def apply_pca(self, features):
        pca = PCA(n_components=self.pca_components)
        return pca.fit_transform(features)

    # Function that runs everything
    def process(self):
        print("\nExtracting features using Resnet...")
        train_features = self.extract_features(self.train_loader)
        test_features = self.extract_features(self.test_loader)
        print("Train features shape after extraction:", train_features.shape)
        print("Test features shape after extraction:", test_features.shape)

        print("\nReducing features with PCA...")
        train_features_pca = self.apply_pca(train_features)
        test_features_pca = self.apply_pca(test_features)
        print("Train features shape after PCA:", train_features_pca.shape)
        print("Test features shape after PCA:", test_features_pca.shape)

        return train_features_pca, test_features_pca

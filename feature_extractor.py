import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.decomposition import PCA


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
    def __init__(self, train_class_limit=500, val_class_limit=100, batch_size=32, pca_components=50):
        self.train_class_limit = train_class_limit
        self.val_class_limit = val_class_limit
        self.batch_size = batch_size
        self.pca_components = pca_components

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                                       transform=self.transform)
        self.val_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                                     transform=self.transform)

        self.filtered_train_data = filter_by_class_limit(self.train_data, self.train_class_limit)
        self.filtered_val_data = filter_by_class_limit(self.val_data, self.val_class_limit)

        self.train_loader = DataLoader(self.filtered_train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.filtered_val_data, batch_size=self.batch_size, shuffle=False)

        self.model = resnet18()
        self.model.fc = nn.Identity()
        self.model.eval()

    def extract_features(self, data_loader):
        features = []
        with torch.no_grad():
            for imgs, _ in data_loader:
                features.append(self.model(imgs))
        return torch.cat(features)

    def apply_pca(self, features):
        pca = PCA(n_components=self.pca_components)
        return pca.fit_transform(features)

    def process(self):
        train_features = self.extract_features(self.train_loader)
        val_features = self.extract_features(self.val_loader)

        train_features_pca = self.apply_pca(train_features)
        val_features_pca = self.apply_pca(val_features)

        return train_features_pca, val_features_pca

import torch
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# Filter function:
# Receives the dataset and filter the images based on the limit
# Basically, it identifies the class, checks the dictionary to see if its under the limit, if so, add the image
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
        self.test_loader = None
        self.train_loader = None
        self.filtered_test_data = None
        self.filtered_train_data = None
        self.model = None
        self.device = None
        self.test_data = None
        self.train_data = None
        self.train_class_limit = train_class_limit
        self.test_class_limit = test_class_limit
        self.batch_size = batch_size
        self.pca = PCA(n_components=pca_components)

    def load_cifar10(self):
        # Prepare to resize images to 224x224x3
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load CIFAR10 with the resize option
        self.train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        self.test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Filter the data to get just the number of the images defined on the constructor
    def filter_cifar10(self):
        print(f"\nFiltering data to get {self.train_class_limit} training images and {self.test_class_limit} testing images (per class)...")
        self.filtered_train_data = filter_by_class_limit(self.train_data, self.train_class_limit)
        self.filtered_test_data = filter_by_class_limit(self.test_data, self.test_class_limit)
        print("Number of training images:", len(self.filtered_train_data))
        print("Number of test images:", len(self.filtered_test_data))

    # Creating with just the images we want
    def create_dataloaders(self):
        self.train_loader = DataLoader(self.filtered_train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(self.filtered_test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    # Initializing ResNet18 Model to extract the feature vectors
    def init_resnet18(self):
        self.model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    # Function used to extract the features
    def extract_features(self, dataloader):
        features = []
        labels = []

        with torch.no_grad():
            for images, lbs in tqdm(dataloader, desc="Extracting Features"):
                images = images.to(self.device)
                lbs = lbs.to(self.device)

                feature = self.model(images)
                feature = feature.view(feature.size(0), -1)

                features.append(feature)
                labels.append(lbs)

        features = torch.cat(features, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()

        return features, labels

    def get_features_and_labels(self):
        self.load_cifar10()
        self.filter_cifar10()
        self.create_dataloaders()
        # self.init_resnet18()
        #
        # print("\nExtracting features using Resnet...")
        # train_features, train_labels = self.extract_features(self.train_loader)
        # test_features, test_labels = self.extract_features(self.test_loader)
        #
        # print("\nReducing features with PCA...")
        # train_features_pca = self.pca.fit_transform(train_features)
        # test_features_pca = self.pca.transform(test_features)
        # print("Train features shape after PCA:", train_features_pca.shape)
        # print("Test features shape after PCA:", test_features_pca.shape)
        #
        # return train_features_pca, train_labels, test_features_pca, test_labels

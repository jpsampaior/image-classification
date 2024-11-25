import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from vgg11 import *


# Definir a classe personalizada para filtrar as imagens
class CustomCIFAR10(Dataset):
    def __init__(self, dataset, train=True, n_train=500, n_test=100):
        self.dataset = dataset
        self.train = train
        self.n_train = n_train
        self.n_test = n_test

        # Filtrar as imagens por classe
        self.data, self.labels = self.filter_data()

    def filter_data(self):
        data = []
        labels = []

        # Para cada classe (0 a 9 no CIFAR-10)
        for label in range(10):
            class_data = []
            class_labels = []

            # Filtra as imagens para a classe
            for i in range(len(self.dataset)):
                if self.dataset.targets[i] == label:
                    class_data.append(self.dataset.data[i])
                    class_labels.append(self.dataset.targets[i])

            # Dividir em treino e teste
            if self.train:
                data.extend(class_data[:self.n_train])  # Pegue as 500 primeiras imagens para treino
                labels.extend(class_labels[:self.n_train])
            else:
                data.extend(class_data[self.n_train:self.n_train + self.n_test])  # Pegue as 100 imagens para teste
                labels.extend(class_labels[self.n_train:self.n_train + self.n_test])

        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # Converter a imagem para tensor e aplicar transformações
        img = transforms.ToTensor()(img)
        return img, label


# Passos para carregar os dados CIFAR-10 e aplicar o filtro
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalização para imagens RGB
])

# Carregar o dataset CIFAR-10 completo
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Criar datasets filtrados
train_dataset = CustomCIFAR10(full_train_dataset, train=True, n_train=500, n_test=100)
test_dataset = CustomCIFAR10(full_test_dataset, train=False, n_train=500, n_test=100)

# DataLoader para treino e teste
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Verificar a quantidade de dados
print(f"Número de imagens de treino: {len(train_loader.dataset)}")
print(f"Número de imagens de teste: {len(test_loader.dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG11().to(device)
train_vgg11_model(model, train_loader, device)
predictions, true_labels = predict_vgg11_model(model, test_loader, device)
accuracy = get_accuracy_vgg11_model(predictions, true_labels)
print(accuracy)


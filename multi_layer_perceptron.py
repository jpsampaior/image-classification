import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MultiLayerPerceptron(nn.Module):
    def __init__(self, features_size=50, hidden_layer_sizes=512, output_size=10, depth=3):
        super(MultiLayerPerceptron, self).__init__()
        # Add first layer - Linear(50, 512) - ReLU
        layers = [nn.Linear(features_size, hidden_layer_sizes), nn.ReLU()]

        # Add middle layers - Linear(512, 512) - BatchNorm(512) - ReLU
        for i in range(depth - 2):
            layers.append(nn.Linear(hidden_layer_sizes, hidden_layer_sizes))
            layers.append(nn.BatchNorm1d(hidden_layer_sizes))
            layers.append(nn.ReLU())

        # Add last layer - Linear(512, 10)
        layers.append(nn.Linear(hidden_layer_sizes, output_size))

        # Connecting all the layers
        self.network = nn.Sequential(*layers)
        self.predicted = None
        self.accuracy = None

    def forward(self, x):
        return self.network(x)

    def train_model(self, train_features, train_labels, num_epochs=10, momentum=0.9):
        if not isinstance(train_features, torch.Tensor):
            train_features = torch.tensor(train_features, dtype=torch.float32)
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels, dtype=torch.long)

        # Using the cross-entropy loss and sgd optimizer with the desired momentum
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), momentum=momentum)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self(train_features)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

    def predict(self, test_features):
        # Convert to tensors if needed
        if not isinstance(test_features, torch.Tensor):
            test_features = torch.tensor(test_features, dtype=torch.float32)

        with torch.no_grad():
            outputs = self(test_features)
            outputs = outputs.numpy()

            self.predicted = np.argmax(outputs, axis=1)

        return self.predicted

    def get_accuracy(self, labels):
        self.accuracy = np.mean(self.predicted == labels)
        return self.accuracy

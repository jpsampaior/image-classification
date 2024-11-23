import torch
import torch.nn as nn
import torch.optim as optim


class MultiLayerPerceptron(nn.Module):
    def __init__(self, features_size=50, hidden_layer_sizes=512, output_size=10, depth=3):
        super(MultiLayerPerceptron, self).__init__()

        # Add first layer - Linear(50, 512)- ReLU
        layers = [nn.Linear(features_size, hidden_layer_sizes), nn.ReLU()]

        # Add middle layers - Linear(512, 512)- BatchNorm(512)- ReLU
        for i in range(depth - 2):
            layers.append(nn.Linear(hidden_layer_sizes, hidden_layer_sizes))
            layers.append(nn.BatchNorm1d(hidden_layer_sizes))
            layers.append(nn.ReLU())

        # Add last layer - Linear(512, 10)
        layers.append(nn.Linear(hidden_layer_sizes, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def train_model(self, train_features, train_labels, num_epochs=10, learning_rate=0.01, momentum=0.9):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        for epoch in range(num_epochs):
            running_loss = 0.0

            optimizer.zero_grad()
            outputs = self(train_features)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    def predict(self, test_features, test_labels):
        correct = 0
        total = 0

        with torch.no_grad():
            outputs = self(test_features)
            _, predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

        accuracy = correct / total
        return accuracy

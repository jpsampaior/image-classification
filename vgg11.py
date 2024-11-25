import torch
import torch.nn as nn
from torch import optim
import numpy as np


# Low epoch number because my computer is weak, it takes a lot to run
def train_vgg11_model(model, train_loader, device, num_epochs=20):
    # Using the cross-entropy and sgd as the pdf says
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            # sending to the device in case of gpu to make faster
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # reseting the grad of previous iterations
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # getting how much the predictions are wrong

            # machine: the more I change this weight, the better I get?
            loss.backward()

            # get the grads to adjust the weights
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total}%")


def predict_vgg11_model(model, test_loader, device):
    predictions = []
    labels = []

    model.eval()

    # using the same logic as I did to extract the features
    with torch.no_grad():
        for inputs, lbs in test_loader:
            inputs = inputs.to(device)
            lbs = lbs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.append(preds)
            labels.append(lbs)

    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    return predictions, labels


def get_accuracy_vgg11_model(predictions, true_labels):
    predictions = np.array(predictions)
    labels = np.array(true_labels)

    return np.mean(predictions == labels)


class VGG11(nn.Module):
    def __init__(self, kernel_size=3):
        super(VGG11, self).__init__()

        self.conv_layers = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=1, padding=kernel_size//2),  # Ajuste dinâmico do padding
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3
            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 4
            nn.Conv2d(256, 256, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 5
            nn.Conv2d(256, 512, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 6
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 7
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 8
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_input_size = None

        self.fc_layers = nn.Sequential(

            # 9
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 10
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 11
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        # passing the data through the conv layers to extract the information
        x = self.conv_layers(x)

        # adjusting the input size based on the tensor dimensions
        if self.fc_input_size is None:
            self.fc_input_size = x.size(1) * x.size(2) * x.size(3)
            self.fc_layers[0] = nn.Linear(self.fc_input_size, 4096).to(x.device)

        # flattening the tensor before sending to the next layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

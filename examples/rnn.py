import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import flor
from flor import MTK as Flor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flor.log("device", str(device))

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="../../data/", train=True, transform=transforms.ToTensor(), download=True
)
flor.log("train_dataset", str(type(train_dataset)))

test_dataset = torchvision.datasets.MNIST(
    root="../../data/", train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(  # type: ignore
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(  # type: ignore
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# Recurrent neural network (many-to-one)
model = RNN(input_size, hidden_size, num_layers, num_classes, device).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Flor.checkpoints(model, optimizer)

# Train the model
total_step = len(train_loader)
for epoch in Flor.loop(range(num_epochs)):
    flor.log("learning_rate", optimizer.param_groups[0]["lr"])
    for i, (images, labels) in Flor.loop(enumerate(train_loader)):  # type: ignore
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    i + 1,
                    total_step,
                    flor.log("loss", float(loss.item())),
                )
            )

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(
        "Test Accuracy of the model on the 10000 test images: {} %".format(
            flor.log("accuracy", 100 * correct / total)
        )
    )

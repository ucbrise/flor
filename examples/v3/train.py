import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torchdata

from random import randint
import numpy as np

import flordb as flor

# Device configuration
device = torch.device(
    flor.arg(
        "device",
        (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
)

seed = flor.arg("seed", default=randint(1, 10000))
torch.manual_seed(seed)

# Hyper-parameters
input_size = 784
hidden_size = flor.arg("hidden", default=500)
num_classes = 10
num_epochs = flor.arg("epochs", 5)
batch_size = flor.arg("batch_size", 32)
learning_rate = flor.arg("lr", 1e-3)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="../data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="../data", train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = torchdata.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torchdata.DataLoader(dataset=test_dataset, batch_size=batch_size)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)  # move ReLU here
        out = self.fc2(out)  # return raw logits for CrossEntropyLoss
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_val_loader(fraction=0.2):
    indices = list(range(len(test_dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(fraction * len(test_dataset)))
    subset_indices = indices[:split]
    sampler = torchdata.SubsetRandomSampler(subset_indices)
    return torchdata.DataLoader(test_dataset, sampler=sampler, batch_size=batch_size)


def validate(val_loader: torchdata.DataLoader):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return int(correct), int(total)


print_every = flor.arg("print_every", 500)

for epoch in flor.loop("epoch", range(num_epochs)):
    for i, (images, labels) in flor.loop("step", enumerate(train_loader)):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_every == 0:
            flor.log("loss", loss.item())

    correct, total = validate(get_val_loader())
    flor.log("val_acc", 100 * correct / total)

    # flor keeps this run's copy of the file.
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        "ckpt.pth",
    )


correct, total = validate(test_loader)
flor.log("accuracy", 100 * correct / total)
flor.log("correct", correct)

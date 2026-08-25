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

# Adaptive-checkpoint throttle. Lower it (e.g. 0) for short runs / demos where
# you want every epoch's torch.save mirrored to .flor/obj_store/; keep the
# default (60s) for long training runs to bound disk usage.
flor.set_ckpt_interval(flor.arg("ckpt_interval_s", 0.0))

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
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Resume from a checkpoint if one exists. On a forward run this lets the user
# continue an interrupted training; on a replay run flor redirects torch.load
# to the matching .flor/obj_store/<ts>/ mirror, so the model jumps to the
# correct historical state instead of starting from random init.
import os as _os

if _os.path.exists("ckpt.pth"):
    _resume = torch.load("ckpt.pth")
    model.load_state_dict(_resume["model"])
    optimizer.load_state_dict(_resume["optimizer"])


def get_val_loader(fraction=0.2):
    indices = list(range(len(test_dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(fraction * len(test_dataset)))
    subset_indices = indices[:split]
    sampler = torchdata.SubsetRandomSampler(subset_indices)
    return torchdata.DataLoader(test_dataset, sampler=sampler, batch_size=batch_size)


def validate(val_loader: torchdata.DataLoader):
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

# v4: gradual typing. Both loops are flor.loop -- inner "step" can be
# skipped/narrowed by flor during replay (`--replay_flor step=...`). Per-iter
# wall times are summarized at loop exit (one time::iter mean + std + n per
# loop scope) rather than logged per step. Plain `for` still works (see the
# inner loop in v3) -- you just lose the replay-narrowing hint for that scope.
# No `with flor.checkpointing(...):` block: per-epoch torch.save below is
# piggy-backed into .flor/obj_store/ automatically.
for epoch in flor.loop("epoch", range(num_epochs)):
    for i, (images, labels) in flor.loop("step", enumerate(train_loader)):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_every == 0:
            flor.log("loss", loss.item())

    correct, total = validate(get_val_loader())
    flor.log("val_acc", 100 * correct / total)
    flor.log("model_norm", sum(p.norm().item() for p in model.parameters()))

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        "ckpt.pth",
    )


correct, total = validate(test_loader)
flor.log("accuracy", 100 * correct / total)
flor.log("correct", correct)

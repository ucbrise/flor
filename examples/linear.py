import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import flor


class Net(nn.Module):
    def __init__(self):
        torch.manual_seed(flor.pin("netseed", random.randint(0, 9999)))
        super(Net, self).__init__()

        self.inpt_dim = 28
        self.agg_factor = 2
        self.num_classes = 10

        self.pool = nn.MaxPool2d(self.agg_factor, self.agg_factor)
        self.fc1 = nn.Linear((self.inpt_dim // self.agg_factor) ** 2, self.num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, 1, (self.inpt_dim // self.agg_factor) ** 2)
        x = self.fc1(x)
        return F.relu(x).view(-1, self.num_classes)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(
    root="./mnist", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, num_workers=4)  # type: ignore

testset = torchvision.datasets.MNIST(
    root="./mnist", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, num_workers=4)  # type: ignore


def eval(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % flor.pin("acc", accuracy)
    )


net = Net()
if torch.cuda.is_available():
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
flor.checkpoints(net, optimizer)  #type: ignore

for epoch in flor.loop(range(2)): #type: ignore
    running_loss = 0.0
    for i, data in flor.loop(enumerate(trainloader, 0)): #type: ignore
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, flor.pin("avg_loss", running_loss / 2000))
                )
                running_loss = 0.0

    if flor.SkipBlock.step_into("training_loop", probed=False):
    flor.SkipBlock.end(net, optimizer)
    eval(net)

print("Finished Training")

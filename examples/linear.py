import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import flor


class Net(nn.Module):
    def __init__(self):
        torch.manual_seed(1217)
        super(Net, self).__init__()

        self.inpt_dim = 28
        self.hidden_layer = 96
        self.agg_factor = 2
        self.num_classes = 10

        self.pool = nn.MaxPool2d(self.agg_factor, self.agg_factor)
        self.fc1 = nn.Linear((self.inpt_dim // self.agg_factor)**2, self.hidden_layer)
        
        self.fc2 = nn.Linear(self.hidden_layer, self.num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, 1, (self.inpt_dim // self.agg_factor) ** 2)        
        x = self.fc1(x)
        x = self.fc2(x)
        return F.relu(x).view(-1, self.num_classes)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(
    root="./mnist", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=4)

testset = torchvision.datasets.MNIST(
    root="./mnist", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=4)


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

    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )


net = Net()
if torch.cuda.is_available():
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in flor.it(range(2)):
    running_loss = 0.0
    if flor.SkipBlock.step_into("training_loop", probed=False):
        for i, data in enumerate(trainloader, 0):
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
            if i % 200 == 199:  # print every 200 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    flor.SkipBlock.end(net, optimizer)
    eval(net)

print("Finished Training")

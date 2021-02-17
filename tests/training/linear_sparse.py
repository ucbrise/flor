import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import flor
import time


class Net(nn.Module):
    def __init__(self):
        torch.manual_seed(1217)
        super(Net, self).__init__()
        self.fc1 = nn.Linear(196, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(4, -1)
        x = F.relu(self.fc1(x))
        return x


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, num_workers=2)

testset = torchvision.datasets.MNIST(root='./mnist', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, num_workers=2)


def eval(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in flor.it(range(80)):
    print(f'epoch: {epoch}')
    running_loss = 0.0
    if flor.SkipBlock.step_into('training_loop', probed=True):
        print('foo')
        time.sleep(0.002)
    flor.SkipBlock.end(net, optimizer)
    eval(net)

print('Finished Training')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):

    def __init__(self):
        try:
            flor.namespace_stack.new()
            torch.manual_seed(1217)
            super(Net, self).__init__()
            self.fc1 = nn.Linear(196, 10)
            flor.namespace_stack.test_force(self.fc1, 'self.fc1')
            self.pool = nn.MaxPool2d(2, 2)
            flor.namespace_stack.test_force(self.pool, 'self.pool')
        finally:
            flor.namespace_stack.pop()

    def forward(self, x):
        try:
            flor.namespace_stack.new()
            x = self.pool(x)
            flor.namespace_stack.test_force(x, 'x')
            x = x.view(4, -1)
            flor.namespace_stack.test_force(x, 'x')
            x = F.relu(self.fc1(x))
            flor.namespace_stack.test_force(x, 'x')
            return x
        finally:
            flor.namespace_stack.pop()


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize
    ((0.5,), (0.5,))])
flor.namespace_stack.test_force(transform, 'transform')
trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=
    True, transform=transform)
flor.namespace_stack.test_force(trainset, 'trainset')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, num_workers=2
    )
flor.namespace_stack.test_force(trainloader, 'trainloader')
testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=
    True, transform=transform)
flor.namespace_stack.test_force(testset, 'testset')
testloader = torch.utils.data.DataLoader(testset, batch_size=4, num_workers=2)
flor.namespace_stack.test_force(testloader, 'testloader')


def eval(net):
    try:
        flor.namespace_stack.new()
        correct = 0
        flor.namespace_stack.test_force(correct, 'correct')
        total = 0
        flor.namespace_stack.test_force(total, 'total')
        with torch.no_grad():
            flor.skip_stack.new(0, 0)
            for data in testloader:
                images, labels = data
                flor.namespace_stack.test_force(images, 'images')
                flor.namespace_stack.test_force(labels, 'labels')
                outputs = net(images)
                flor.namespace_stack.test_force(outputs, 'outputs')
                _, predicted = torch.max(outputs.data, 1)
                flor.namespace_stack.test_force(_, '_')
                flor.namespace_stack.test_force(predicted, 'predicted')
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            flor.skip_stack.pop()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    finally:
        flor.namespace_stack.pop()


net = Net()
flor.namespace_stack.test_force(net, 'net')
criterion = nn.CrossEntropyLoss()
flor.namespace_stack.test_force(criterion, 'criterion')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
flor.namespace_stack.test_force(optimizer, 'optimizer')
flor.skip_stack.new(3, 0)
for epoch in range(2):
    running_loss = 0.0
    flor.namespace_stack.test_force(running_loss, 'running_loss')
    flor.skip_stack.new(2)
    if flor.skip_stack.peek().should_execute(True):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            flor.namespace_stack.test_force(inputs, 'inputs')
            flor.namespace_stack.test_force(labels, 'labels')
            optimizer.zero_grad()
            outputs = net(inputs)
            flor.namespace_stack.test_force(outputs, 'outputs')
            loss = criterion(outputs, labels)
            flor.namespace_stack.test_force(loss, 'loss')
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, 
                    running_loss / 2000))
                running_loss = 0.0
                flor.namespace_stack.test_force(running_loss, 'running_loss')
    _, running_loss = flor.skip_stack.pop().proc_side_effects(optimizer,
        running_loss)
    eval(net)
flor.skip_stack.pop()
print('Finished Training')

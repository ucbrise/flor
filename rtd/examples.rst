Flor Examples
===============


scikit learn logger
--------------------
Consider the following snippet of code that loads, pre-processes, trains, and tests a random forest classifier.

``
import flor
import pandas as pd

import cloudpickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


log = flor.log #setting a log variable for ease of use

@flor.track #we want to track this function
def train_model(n_estimators, X_tr, y_tr):
    clf = RandomForestClassifier(n_estimators=log.param(n_estimators)).fit(X_tr, y_tr)
    with open(log.write('clf.pkl'), 'wb') as classifier:
        cloudpickle.dump(clf, classifier)
    return clf

@flor.track
def test_model(clf, X_te, y_te):
    score = log.metric(clf.score(X_te, y_te)) #has flor log our score

@flor.track
def main(x, y, z):
    # Load the Data
    movie_reviews = pd.read_json(log.read('data.json'))

    movie_reviews['rating'] = movie_reviews['rating'].map(lambda x: 0 if x < z else 1)

    # Do train/test split-
    X_tr, X_te, y_tr, y_te = train_test_split(movie_reviews['text'], movie_reviews['rating'],
                                              test_size=log.param(x),
                                              random_state=log.param(y))

    # Vectorize the English sentences
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_tr)
    X_tr = vectorizer.transform(X_tr)
    X_te = vectorizer.transform(X_te)

    # Fit the model
    for i in [1, 5]:
        clf = train_model(i, X_tr, y_tr)
        test_model(clf, X_te, y_te)

    the_answer_to_everything = log.param(42)


with flor.Context('basic'):
    main(0.2, 92, 5)


``


Pytorch Neural Network Logging
--------------------------------
In this example we take a neural network built using pytorch and add flor logging.

``
#Source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import flor
log = flor.log

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

@flor.track #we want to track what is going on inside this function
def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = NeuralNet(log.param(input_size), log.param(hidden_size), log.param(num_classes)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=log.param(learning_rate))

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                log.metric(epoch) #here we log the epoch, iteration number, and loss
                log.metric(i)
                log.metric(loss.item())

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = 100 * (correct / total)
            log.metric(acc) #here we are logging the accuracy at every iteration

    print('Accuracy of the network on the 10000 test images: {} %'.format(acc))


with flor.Context('pytorch_demo_nn'):
    main()

``

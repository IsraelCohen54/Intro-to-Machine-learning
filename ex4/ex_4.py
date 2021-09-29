# 205812290 313369183

import torch
import sys
import numpy as np
import torchvision
from torch import nn,optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import math #for using floor to part the data from fashionMnist
from torchvision import transforms, utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ModelAB(nn.Module):
    def __init__(self, image_size):
        super(ModelAB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x1 = x.view(-1,self.image_size)
        x2 = F.relu(self.fc0(x1))
        x3 = F.relu(self.fc1(x2))
        x4 = self.fc2(x3)
        return F.log_softmax(x4, dim=1)

class ModelC(nn.Module):
    def __init__(self, image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size,100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50,10)
    def forward(self, x):
        dropout = torch.nn.Dropout(p=0.2)
        x1 = x.view(-1,self.image_size)
        x2 = F.relu(self.fc0(x1))
        x2 = dropout(x2)
        x3 = F.relu(self.fc1(x2))
        x3 = dropout(x3)
        x4 = self.fc2(x3)
        return F.log_softmax(x4, dim=1)

class ModelD(nn.Module):
    def __init__(self, image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x1 = x.view(-1, self.image_size)
        batch = nn.BatchNorm1d(784)
        x1 = batch(x1)
        x2 = F.relu(self.fc0(x1))
        batch1 = nn.BatchNorm1d(100)
        x2 = batch1(x2)
        x3 = F.relu(self.fc1(x2))
        x4 = self.fc2(x3)
        return F.log_softmax(x4, dim=1)

class ModelE(nn.Module):
    def __init__(self, image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
    def forward(self, x):
        x1 = x.view(-1,self.image_size)
        x2 = F.relu(self.fc0(x1))
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = F.relu(self.fc3(x4))
        x6 = F.relu(self.fc4(x5))
        x7 = self.fc5(x6)
        return F.log_softmax(x7, dim=1)

class ModelF(nn.Module):
    def __init__(self, image_size):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
    def forward(self, x):
        x1 = x.view(-1,self.image_size)
        x2 = F.sigmoid(self.fc0(x1))
        x3 = F.sigmoid(self.fc1(x2))
        x4 = F.sigmoid(self.fc2(x3))
        x5 = F.sigmoid(self.fc3(x4))
        x6 = F.sigmoid(self.fc4(x5))
        x7 = self.fc5(x6)
        return F.log_softmax(x7, dim=1)

"""
def train(model,optimizer):
    model.train()
    counter = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        #loss = F.nll_loss(output, labels.type(torch.int64))
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))
"""
def pytorch_accuracy(y_pred, y_true):

    y_pred = y_pred.argmax(1)
    return (y_pred == y_true).float().mean() * 100

def train(model,optimizer):
    model.train()
    trainLoss=0.0
    acc_sum=0.0
    example_counter = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data).double()
        loss = F.nll_loss(output, labels)
        #loss = F.nll_loss(output, labels.type(torch.int64))
        trainLoss += loss.item()
        loss.backward()
        optimizer.step()
        #pred=output.max(1, keepdim=True)[1]
        acc_sum += float(pytorch_accuracy(output, labels)) * len(data)
        example_counter += len(data)
    trainLoss /= len(train_loader)
    accuracy = acc_sum / example_counter
    return trainLoss, accuracy
        # print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #            100. * batch_idx / len(train_loader), loss.item()))
    print("trainloss= ", trainLoss, "accuracy= ", accuracy,"\n")


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum().item()
    test_loss /= len(valid_loader.dataset)
    return test_loss, 100*correct/len(valid_loader.dataset)
    print(
        f'=========================== lr\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(valid_loader.dataset)} '
        f'({100. * correct / len(valid_loader.dataset):.0f}%)\n')

"""def test_val(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(train_loader.dataset)
    print(
        f'=========================== lr\nTest_val set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} '
        f'({100. * correct / len(train_loader.dataset):.0f}%)\n')
"""

def modelA():
    model = ModelAB(image_size=28 * 28)
    #model.add_module("model",ModelAB(image_size=28 * 28))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer

def modelB():
    model = ModelAB(image_size=28 * 28)
    #model.add_module()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    return model, optimizer

def modelC():
    model = ModelC(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer

def modelD():
    model = ModelD(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer

def modelE():
    model = ModelE(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    return model, optimizer

def modelF():
    model = ModelF(image_size=28 * 28)
    # model.add_module()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    return model, optimizer

# printing prediction:
def teleTest():
    model.eval()
    res = []
    for data in test_x:
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        [res.append(x) for x in list(pred.numpy().flatten())]
    return res


#main:
if __name__ == '__main__':
    # test_x_data = sys.argv[1] #make a tensor@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # test_x = np.loadtxt(test_x_data, max_rows= 500)
    # test_of_x = torch.tensor(test_x)

    # load the data:
    train_split_percent = 0.8
    batch_size = 32

    transforming = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])

    # training loop:
    full_train_dataset = torchvision.datasets.FashionMNIST('./datasets/', train=True, download=True, transform=transforming)
    test_dataset = torchvision.datasets.FashionMNIST('./datasets/', train=False, download=True, transform=transforming)



    num_data = len(full_train_dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)

    split = math.floor(train_split_percent * num_data)

    train_indices = indices[:split]
    train_dataset = Subset(full_train_dataset, train_indices)

    valid_indices = indices[split:]
    valid_dataset = Subset(full_train_dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    test_x = np.loadtxt(sys.argv[3])
    test_x = test_x.astype(np.float32)
    test_x = transforming(test_x)
    #test_x = DataLoader(test_dataset, batch_size=32)

    #model, optimizer = modelA()
    #model, optimizer = modelB()
    #model, optimizer = modelC()
    model, optimizer = modelD()
    #model, optimizer = modelE()
    #model, optimizer = modelF()

    """#To see the images:
    inputs = next(iter(train_loader))[0]
    input_grid = utils.make_grid(inputs)

    fig = plt.figure(figsize=(10, 10))
    inp = input_grid.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.show()
    """
    # Time to run :)
    # for epoch in range(1, 11):
    #     train(model, optimizer)
    #     #test_val(model)
    #     test(model)
    model.float()
    for epoch in range(10):
        train(model, optimizer)

    ans = teleTest()

    with open('test_y', 'w') as f:
        for i,prediction in enumerate(ans):
            f.write(str(prediction))
            if i != len(ans)-1:
                f.write("\n")


    #__test(self=test_loader)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from __future__ import print_function

import os
import time
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

# In[ ]:
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--save_path', type=str, default='./model')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--display_epoch', type=int, default=1)
parser.add_argument('--schedule_lr', type=str, default='yes')
args = parser.parse_args()

# In[ ]:
# Setting log
# f = open('log.txt', 'a')

# Parameters
DOWNLOAD_CIFAR10 = False
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
x_epoch = []
y_train_loss = []
y_train_acc = []
y_test_acc = []

# Fix random seed
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# In[ ]:

# mkdir save folder
save_dir_path = os.path.join(args.save_path, args.dataset)
os.makedirs(save_dir_path, exist_ok=True)

# Load dataset

if not (os.path.exists('./data')) or not (os.listdir('./data')):
    DOWNLOAD_CIFAR10 = True

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True, 
                                        download=DOWNLOAD_CIFAR10,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=DOWNLOAD_CIFAR10,
                                       transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# In[ ]:
def save_network(network, path, epoch_label):
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)
    torch.save(network.state_dict(), file_path)

def load_network(network, path, epoch_label):
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)
    network.load_state_dict(torch.load(file_path))
    return network

def save_curve():
    plt.title('Train/Test vs Epoch')
    plt.subplot(2, 1, 1)
    plt.plot(x_epoch, y_train_acc, 'bs-', label='train_acc')
    plt.plot(x_epoch, y_test_acc, 'rs-', label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.subplot(2, 1, 2)
    plt.plot(x_epoch, y_train_loss, 'bs-', label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend() # display label
    plt.savefig('result.jpg')
# In[ ]:


# LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    

# In[ ]:


# Train
def train(model, criterion, optimizer, scheduler, num_epochs, device):    
    
    print('Begin Training',
            f"Train number: {len(trainset)}",
            f"Train Epoch: {num_epochs}",
            f"Batch Size: {args.batch_size}",
            f"Train LR: {args.lr}")
    # print('Begin Training',
    #         f"Train number: {len(trainset)}",
    #         f"Train Epoch: {num_epochs}",
    #         f"Batch Size: {args.batch_size}",
    #         f"Train LR: {args.lr}", file=f)

    start = time.time()
    train_loss = 0.0
    corrects = 0
    total = 0

    for epoch in range(1, 1+num_epochs):
        model.train()
        if (args.scheduler_lr):
            scheduler.step()
        
        for images, labels in enumerate(trainloader, 0):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()
            train_loss += loss
            
        if epoch % args.display_epoch == 0:
            x_epoch.append(epoch)
            y_train_loss.append((train_loss/len(trainset)))
            y_train_acc.append((corrects/total))
            print(f"Epoch [{epoch}/{num_epochs}],"
                    f"Training Loss: {(train_loss/len(trainset)):.8f},"
                    f"Training Corrects: {corrects} / {total},"
                    f"Training Accuracy: {(corrects/total):.3f}")

            # print(f"Epoch [{epoch}/{num_epochs}],"
            #         f"Training Loss: {(train_loss/len(trainset)):.8f},"
            #         f"Training Corrects: {corrects} / {total},"
            #         f"Training Accuracy: {(corrects/total):.3f}", file=f)
            train_loss = 0.0
            corrects = 0
            total = 0
            test(model)
        
        # if epoch % 10 == 0:
        #     save_network(model, save_dir_path, str(epoch))
    end = time.time()
    # save_curve()
    print(f"Finished Training, {(end-start)/3600} h")
    # print(f"Finished Training, {(end-start)/3600} h", file=f)
    torch.cuda.empty_cache()


# In[ ]:


# Test
def test(model):

    model.eval()
    corrects = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()

    y_test_acc.append((corrects/total))
    print( f"Testing Corrects: {corrects} / {total},"
            f"Testing Accuracy: {(corrects/total):.3f},")
    # print( f"Testing Corrects: {corrects} / {total},"
    #         f"Testing Accuracy: {(corrects/total):.3f},", file=f)


# In[ ]:


# Start
net = LeNet()
# net = ResNet(Bottleneck, [3, 4, 6, 3], args.num_classes)
# net = load_network(net, save_dir_path, args.which_epoch)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=MOMENTUM , weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on", device)
# print("Run on", device, file=f)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # print("Let's use", torch.cuda.device_count(), "GPUs!", file=f)
    net = nn.DataParallel(net)
    
net = net.to(device)
train(net, criterion, optimizer, scheduler, args.epochs, device)

# f.close()
# In[ ]:





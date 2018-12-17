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
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--display_epoch', type=int, default=1)
parser.add_argument('--scheduler_lr', type=str, default='yes')
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

if not (os.path.exists('../data')) or not (os.listdir('../data')):
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
    plt.plot(x_epoch, y_train_acc, 'bs-', label='train_acc')
    plt.plot(x_epoch, y_test_acc, 'rs-', label='test_acc')
    plt.plot(x_epoch, y_train_loss, 'gs-', label='test_loss')
    plt.xlabel('Epoch')
    plt.legend() # display label
    plt.savefig('result.jpg')
 
# In[ ]:


# BottleNet
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)) # 注意，此处也需要stride
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# In[ ]:


# ResNet-18
class ResNet18(nn.Module):
    def __init__(self, block, num_blocks_arr, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=1) # not 7
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = self._make_layer(block, num_blocks_arr[0], 64, 64, downsample=False)
        self.conv3 = self._make_layer(block, num_blocks_arr[1], 64, 128, downsample=True)
        self.conv4 = self._make_layer(block, num_blocks_arr[2], 128, 256, downsample=True)
        self.conv5 = self._make_layer(block, num_blocks_arr[3], 256, 512, downsample=True)
        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, num_blocks, in_channels, out_channels, downsample):
        layers = []
        if downsample == False:
            layers.append(block(in_channels, out_channels, 1))
            for i in range(num_blocks-1):
                layers.append(block(out_channels, out_channels, 1))
        else:
            layers.append(block(in_channels, out_channels, 2))
            for i in range(num_blocks-1):
                layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers) # *layers
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.max_pool(out) # notice
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
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
        if (args.scheduler_lr == 'yes'):
            scheduler.step()
        
        for images, labels in trainloader:
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
    #save_curve()
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

net = ResNet18(Bottleneck, [2, 2, 2, 2], args.num_classes)
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





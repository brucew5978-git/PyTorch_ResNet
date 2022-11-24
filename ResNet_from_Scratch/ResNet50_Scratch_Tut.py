import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
#CIFAR datasets imported from torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import gc

#Source: https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#Dataloader

def data_loader(data_dir, batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    #Common data normalization values

    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    #Resizes img to 224 by 224 pixels and transform into tensor

    if test:

        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )


        '''dataset = datasets.MNIST(
            root=data_dir, train=False,
            download=True, transform=transform
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        ) '''

        return data_loader


    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform
    )    


    '''train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform
    )'''

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size*num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    #Sampler gets random sample image

    train_loader=torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler
    )

    return(train_loader, test_loader)

#CIFAR10 dataset
train_loader, valid_loader = data_loader(data_dir='./data', batch_size=64)

test_loader = data_loader(data_dir='./data', batch_size=64, test=True)


#Residual Block implementation

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.relu=nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out=self.conv1(x)
        out=self.conv2(out)
        if self.downsample:
            residual=self.downsample(x)

        out+=residual
        #residual is reference added back, its what differentiates ResNet's higher performance against other DNN
        out=self.relu(out)
        return out


#ResNet Implementation

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride !=1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )

        layers=[]
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes=planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.layer0(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        x=self.avgpool(x)
        x=x.view(x.size(0), -1)
        x=self.fc(x)

        return x


#Classifiers

num_classes = 10
num_epochs = 20
batch_size = 16
learning_rate = 0.01

model = ResNet(ResidualBlock, [3,4,6,3]).to(device)
#Conv layer: [3,4,6,3] according to ResNet architecture for ResNet34

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)
#Optimizer values can be changed for varying results

total_step = len(train_loader)

import matplotlib.pyplot as plt
#Training

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #move tensors to the configured device
        images= images.to(device)
        labels = labels.to(device)

        '''plt.imshow(images.view(28,28))
        plt.show()'''
        #Forward pass
        outputs=model(images)
        #Selected model used to predict the labels
        loss=criterion(outputs, labels)


        #Backward and optimize
        optimizer.zero_grad()
        #Need to clear vector gradient in python or it accumulates into next iteration of calcs
        loss.backward()
        #Perform back probagation
        optimizer.step()
        #Update layer weights according to losses
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, loss.item()))

    #Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            _, predicted = torch.max(outputs.data, 1)
            total+=labels.size(0)
            correct +=(predicted==labels).sum().item()
            del images, labels, outputs

        print('Accuracy of network on the {} validation images: {} %'.format(5000, 100 * correct / total))
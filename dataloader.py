import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import numpy as np


def get_data_loader(name, batch_size = 128, workers=4):
    normalize1 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    normalize2 = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))#OALL,MD
    normalize3 = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))#Mres
    if name == 'cifar10':

        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize2,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize2,
        ])),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)

    elif name == 'cifar10_extra':

        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize2,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize2,
        ])),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)


    elif name == 'cifar100':
        train_transform = transforms.Compose([ transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(),normalize2,])
        val_transform = transforms.Compose([ transforms.ToTensor(),normalize2,])
        train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=val_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    elif name == 'LSUN':
        transform = transforms.Compose([ transforms.ToTensor(),normalize2,])
        val_dataset = torchvision.datasets.ImageFolder("./data/LSUN_resize", transform=transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         shuffle=False, num_workers=2) 
        train_loader = val_loader
    elif name == 'Tiny':
        transform = transforms.Compose([ transforms.ToTensor(),normalize2,])
        val_dataset = torchvision.datasets.ImageFolder("./data/Imagenet_resize", transform=transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                         shuffle=False, num_workers=2) 
        train_loader = val_loader
    else:
        print('undefined dataset')
        return -1
    return train_loader, val_loader

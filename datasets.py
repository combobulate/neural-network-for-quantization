#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:21:57 2019

@author: Zefyr Scott

This contains the loadDataset function only. See the readme for more info.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from utee import selector

def loadDataset(dataset, encoder, classifier, batch_size_train, batch_size_test):
    pretrained_classifier = None
    
    if (dataset == 'MNIST'):
        classes = 10
        unencoded_dims = { 'height': 28,
                          'width': 28,
                          'size': 28*28} #28*28 images
        train_dataset = torchvision.datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307, ), (0.1307, ))
                        ]))
        test_dataset = torchvision.datasets.MNIST(
                root='./data',
                train=False,
                download=True,
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307, ), (0.1307, ))
                        ]))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size_train,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size_test,
                                                  shuffle=False)
        if (classifier == 'Pretrained'):
            pretrained_classifier, _, _ = selector.select('mnist')
    
    elif (dataset == 'CIFAR10'):
        classes = 10
        if (encoder in ('AlexNetEncode', 'ResNetEncode')):
            unencoded_dims = { 'height': 224,
                          'width': 224,
                          'size': 224*224*3} # Resized RGB images
            train_dataset = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
            test_dataset = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
        else:
            unencoded_dims = { 'height': 32,
                      'width': 32,
                      'size': 32*32*3} # 32*32 RGB images
            train_dataset = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
            test_dataset = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size_train,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size_test,
                                                  shuffle=False)
        if (classifier == 'Pretrained'):
            pretrained_classifier, _, _ = selector.select('cifar10')

    return (train_loader, test_loader), classes, unencoded_dims, pretrained_classifier

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:47:30 2019

@author: Zefyr Scott

Adapted from
https://medium.com/@athul929/hand-written-digit-classifier-in-pytorch-42a53e92b63e

This is the functions used for driving the testing and training activity.
See the readme for more info.
"""

import torch
import torchvision
import supermodel as sup
import tools
from datetime import datetime
import matplotlib.pyplot as plt

def testandtrain(loaders,
            num_epochs,
            learning_rate,
            momentum,
            unencoded_dims,
            subsets,
            dims,
            mse_multiplier,
            encoded_size,
            classes,
            encoder_hidden_size,
            classifier_hidden_size,
            loss_function,
            splitter,
            encoder,
            decoder,
            classifier,
            outfile,
            break_after_first,
            train_only,
            classify_only, 
            print_hooks,
            using_notebook,
            device,
            pretrained_classifier,
            print_status):
    
    models = sup.Models(subsets, dims, encoded_size, encoder_hidden_size, unencoded_dims,
                        classes, classifier_hidden_size, splitter, encoder, decoder,
                        classifier, classify_only, pretrained_classifier)
    models.to(device)
    
    # Train function calls test function after each epoch, for running analysis
    train_accuracy, test_accuracy_unrounded, test_accuracy_rounded = train(models, loaders, num_epochs, 
                        learning_rate, momentum, mse_multiplier, subsets, loss_function, outfile, break_after_first,
                        train_only, classify_only, print_hooks, using_notebook, device, print_status)
    
    if(train_only or break_after_first):
        return "n/a", "n/a"
    else:
        return abs(train_accuracy - test_accuracy_unrounded), abs(train_accuracy - test_accuracy_rounded)

def train(models, loaders, num_epochs, learning_rate, momentum, mse_multiplier, subsets,
          loss_function, outfile, break_after_first, train_only, classify_only,
          print_hooks, using_notebook, device, print_status):

    lossf = sup.LossFunction(loss_function, mse_multiplier, subsets)

# Uncomment the below for verification of parameters
#    for name, param in models.named_parameters():
#        if param.requires_grad:
#            print (name)
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)
#    optimizer = torch.optim.SGD(models.parameters(), lr=learning_rate, momentum=momentum)
    
    first_time = datetime.now()
    
    printout = tools.readyStatus(using_notebook, print_status)
    
    for epoch in range(num_epochs):
        train_accuracy, test_accuracy_unrounded, test_accuracy_rounded  = train_epoch(models, loaders, epoch, num_epochs, device, subsets, lossf, outfile, break_after_first, train_only, classify_only, print_hooks, using_notebook, printout, optimizer, first_time, print_status)

    return train_accuracy, test_accuracy_unrounded, test_accuracy_rounded 

def train_epoch(models, loaders, epoch, num_epochs, device, subsets, lossf, outfile, break_after_first, train_only, classify_only, print_hooks, using_notebook, printout, optimizer, first_time, print_status):
    correct = 0
    total = 0
    train_loader, test_loader = loaders
    total_step = len(train_loader)
    models.train()
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        sample_loss, mse_sample_loss, cel_sample_loss, predicted, label_total, encodedvector, encodedsample, decodedsample = train_loop(images, labels, device, models, outfile, classify_only, print_hooks, lossf, subsets, optimizer)

        if (i+1) % 100 == 0:
            with open(outfile, 'a') as outf:
                print(datetime.now().time(),
                  'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, CELLoss: {:.4f}, MSELoss: {:.4f}' .format(epoch+1,
                         num_epochs, i+1, total_step, sample_loss, cel_sample_loss, mse_sample_loss), file=outf)
            if(not classify_only):
                if(len(encodedsample[0].size()) == 4): ## expects color layers of images which can be rejoined and displayed
                    tools.imshow(torchvision.utils.make_grid(tools.showsplitrgblayers(encodedsample), padding = 0))
                    tools.imshow(torchvision.utils.make_grid(decodedsample[0]))

                plt.hist(encodedvector.cpu().numpy().flatten(), bins=50)
                plt.title('Epoch [{}/{}], Step [{}/{}]'.format(epoch+1, num_epochs, i+1, total_step))
                plt.show()

        total += label_total
        correct += (predicted==labels).sum().item()
        
        remaining = tools.remaining_time((datetime.now() - first_time) * ((total_step * num_epochs) / ((i+1) + epoch * total_step) - 1))
        
        if(print_status):
            tools.status("\rStatus: Epoch {} of {}, batch {} of {}, est {} remaining  ".format(epoch + 1, num_epochs, i + 1, total_step, remaining), printout, using_notebook)

        torch.cuda.empty_cache()
        if (break_after_first): break
        
    train_accuracy = 100 * correct / total
    with open(outfile, 'a') as outf:
        print(datetime.now().time(),
          '\nAccuracy of the network on the training epoch: {} %'.format(train_accuracy), file=outf)
        
    # Test after each epoch, but only the last epoch's test accuracy is returned
    if (not break_after_first and not train_only):
        test_accuracy_unrounded = test(models, test_loader, outfile, classify_only, device)
        test_accuracy_rounded = test(models, test_loader, outfile, classify_only, device, round_output = True)
    else:
        test_accuracy_unrounded, test_accuracy_rounded = None, None
    return train_accuracy, test_accuracy_unrounded, test_accuracy_rounded

def train_loop(images, labels, device, models, outfile, classify_only, print_hooks, lossf, subsets, optimizer):    
    
    encoded, rounded = models(images, "encode", outfile, classify_only, print_hooks)
    
    if(not classify_only):
        encodedvector = torch.cat(encoded, 1).detach().clone()
        encodedsample = []
        for part in encoded:
            encodedsample.append(part.detach().clone())
    else: encodedvector, encodedsample = None, None
    
    out, decodedsample = models(encoded, "decode and classify", outfile, classify_only, print_hooks)
    decodedsample = decodedsample.cpu()
    loss, mseloss, celloss = lossf(out, labels, encoded, rounded, print_hooks, outfile)

    optimizer.zero_grad()
    loss.backward()
#    tools.plot_grad_flow(models.named_parameters())
    optimizer.step()
        
    sample_loss = loss.item()
    mse_sample_loss = mseloss.item() if(mseloss) else 0
    cel_sample_loss = celloss.item() if(celloss) else sample_loss

    _, predicted = torch.max(out.data, 1)
    label_total = labels.size(0)
    return sample_loss, mse_sample_loss, cel_sample_loss, predicted, label_total, encodedvector, encodedsample, decodedsample
   
def test(models, test_loader, outfile, classify_only, device, round_output = False):
    with torch.no_grad():
        correct = 0
        total = 0
        models.eval()
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            predicted, label_total = test_loop(images, labels, device, outfile, classify_only, models, round_output)
            total += label_total
            correct += (predicted==labels).sum().item()
            torch.cuda.empty_cache()
        
        rounding = 'with' if (round_output) else 'without'
        
        with open(outfile, 'a') as outf:
            print(datetime.now().time(),
              '\nAccuracy of the network on the {} test images {} rounding: {} %'.format(total, rounding, 100 * correct / total), file=outf)
        
        test_accuracy = 100 * correct / total
        return test_accuracy
    
def test_loop(images, labels, device, outfile, classify_only, models, round_output):
    encoded, rounded = models(images, "encode", outfile,
            classify_only, round_output=round_output)
    
    out, decodedsample = models(encoded, "decode and classify", outfile, 
            classify_only)
    
    _,predicted = torch.max(out.data, 1)
    label_total = labels.size(0)
    return predicted, label_total
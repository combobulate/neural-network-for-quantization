# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:31:09 2019

@author: Zefyr Scott

Various functions to support the network. See the readme for more info.
"""

import torch
from IPython.display import display
import ipywidgets as widgets
import psutil
import os
import sys
import gc
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D 

try:
    import winsound as output # for alert on completion
except ImportError:
    from google.colab import output # note this will only alert if the tab is active

def alert():
    '''
    Custom stuff for audio alert both locally and on Google colab
    '''
    if hasattr(output, 'SND_ALIAS'):
        output.PlaySound("SystemAsterisk", output.SND_ALIAS)
    else:
        output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')
    return None

def readyStatus(using_notebook, print_status):
    if (using_notebook and print_status):
        printout = widgets.HTML()
        display(printout)
        return printout
    else:
        return None

def status(string, printout, using_notebook):
    if (using_notebook):
        printout.value=string
    else:
       print('\r' + string, end='\r')
    return None

def remaining_time(time):
    '''
    Formats a timedelta object into a string output of the time difference
    '''
    total_seconds = int(time.total_seconds())
    hours, remainder = divmod(total_seconds, 60*60)
    minutes, seconds = divmod(remainder, 60)
    #d = datetime.strptime(str(c), "%H%M%S")
    return "{} hr {} min {} sec".format(hours, minutes, seconds)

def split_sizes(length, sections):
    '''
    Takes a length l (usually will be the number of values in the
    first feature dimension) and a number n of sections to split that length into.
    It outputs a tuple giving the sizes to split that length into, as follows:
       l % n integers of value l // n + 1
       n - l % n integers of value l // n
    For example:
       a = split_sizes(20, 6)
       print(a)
    Output:
       (4, 4, 3, 3, 3, 3)
    This can then be used in torch.split, ie:
       b = torch.randn(3, 20)
       c = b.split(a, 1)
    Note that this is designed such that the torch.split function can be used either
    for custom arbitrary splits or for a specific number of approximately equal splits.
    In this latter usage, torch.split will simply simulate numpy.array_split. If only
    the latter usage is needed, just use the numpy function. Also note that this
    simulates numpy.array_split rather than torch.chunk. Torch.chunk has undesirable
    behavior that may not get fixed:
       https://github.com/pytorch/pytorch/issues/9382
    This is also distinct from the torch.split behavior when passing an integer rather
    than a tuple, in which case torch.split uses that integer as the size of each
    split rather than the number of chunks to split into.
    '''
    output = []
    
    for _ in range(length % sections):
        output.append(length // sections + 1)
    for _ in range(sections - length % sections):
        output.append(length // sections)
    
    return tuple(output)

def split_features(tensor, subsets, dims):
    '''
    Takes in:
        * a tensor of any number of dimensions
        * a tuple of sizes to split a particular dimension into, or a tuple of such
            tuples
        * the corresponding dimension to split along, or a tuple of corresponding
            dimensions
    Outputs a list of the subtensors. For example, to split a batch of CIFAR10 images,
    which are RGB and 32x32 pixels, into two columns and four rows:
    columns = (16, 16)
    rows = (8, 8, 8, 8)
    split_features(images, (rows, columns), (2, 3))
    '''
    # If input is a single subset/dim pair, format as tuples
    if (type(subsets[0]) == int and type(dims) == int):
        subsets = (subsets,)
        dims = (dims,)
    
    # Split along first dimension passed in dims
    tensorsplit = list(tensor.split(subsets[0], dims[0]))

    # Base case is that only one dimension is passed in dims. If more than one,
    # then split each tensor in the tensorsplit list along the next dimension
    # in the tuple.    
    if (len(dims) != 1):
        flatten = []
        for i in range(len(subsets[0])):
            tensorsplit[i] = split_features(tensorsplit[i], subsets[1:], dims[1:])
            for j in range(len(tensorsplit[i])):
                flatten.append(tensorsplit[i][j])
        tensorsplit = flatten
        
    return tensorsplit

def unsplit_features(tensorsplit, subsets, dims):
    '''
    Undoes the actions of split_features to recombine into the original input.
    '''
    # If input is a single subset/dim pair, format as tuples
    if ((type(subsets[0]) == int and type(dims) == int)):
        subsets = (subsets,)
        dims = (dims,)
    
    # Rejoin along last dimension passed in dims. If only one dimension was passed,
    # this is just a list of one
    tensorset = []
    for i in range(0, total_subsets(subsets), len(subsets[-1])):
        for j in range(total_subsets(subsets) // len(subsets[-1])):
            tensorsubset = torch.cat((tensorsplit[ j * len(subsets[-1]) : (j+1) * len(subsets[-1]) ]), dims[-1])
            tensorset.append(tensorsubset)
    
    # If more than one item in the list, keep calling the function
    if (len(dims) != 1):
        return unsplit_features(tensorset, subsets[:-1], dims[:-1])
    # If just one item in the list that's the base case, so return it
    else:
        return tensorset[0]        

def countSubitems(aSet):
    '''
    aSet: Any compound data type, which may contain compound data types
    which may themselves contain further compound data types, etc. IE
    countSubitems([4, (abs, 'b')]) will return 3.
    
    Will actually support being passed any other data type, simply returning
    a count of 1 for anything not compound. So countSubitems(True) will return
    1, if you really want to do that.
    returns: int, how many values are in the compound data type, exclusive
    of the compound data type itself or any such in its contents.
    '''
    if type(aSet) in [list, tuple]:
        return sum([countSubitems(x) for x in aSet])
    elif type(aSet) in [dict]:
        return sum([countSubitems(aSet[x]) for x in aSet])
    else:
        return 1

def total_subsets(subsets):
    if (type(subsets[0]) == int):
        return len(subsets)
    else:
        total = len(subsets[0])
        for i in range(1, len(subsets)):
            total = total * len(subsets[i])
        return total
    
def memReport(outfile):
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            with open(outfile, 'a') as outf:
                print(type(obj), obj.size(), file = outf)
    
def cpuStats(outfile):
    with open(outfile, 'a') as outf:
        print(sys.version, file = outf)
        print(psutil.cpu_percent(), file = outf)
        print(psutil.virtual_memory(), file = outf)  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse, file = outf)

## imshow from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize = (10,2))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def showsplitrgblayers(tensorlist):
    '''Use for displaying separated RGB layers. Expects input to be list of three
    tensors, the results of splitting 4D RGB tensors into 3D tensors for each channel.
    Displays first image in the lists only.
    '''
    dim1, dim2 = tensorlist[0].size()[2], tensorlist[0].size()[3]
    layersep = []
    for i in range(len(tensorlist)):
        newlayer = torch.zeros(3, dim1, dim2)
        newlayer[i] = tensorlist[i][0].detach()
        layersep.append(newlayer)
    return layersep

def show2dsplit(tensorlist, rows, cols):
    '''Use for displaying 4D images separated into a grid. Expects input to be a list
    of tensors, where each entry in each list is a part of this grid. Grid is
    rows x cols. Displays first image in the lists only.
    '''
    f, axarr = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            show = tensorlist[i * rows + j][0] / 2 + 0.5
            npshow = show.numpy()
            axarr[i,j].imshow(np.transpose(npshow, (1, 2, 0)))

## plot_grad_flow from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
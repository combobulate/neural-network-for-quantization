# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:31:07 2019

@author: Zefyr Scott

This contains:
    * Primary models and submodels: used always when the encode, decode, and classify
        functionalities are all used.
    * Various configuration specific modules, whose usage is configured in manager.py
See the readme for more info.
"""
import math
import torch
import torch.nn as nn
import torch.autograd
import torchvision.models as models
import torch.nn.functional as F
import tools

'''
### PRIMARY MODELS AND SUBMODELS ###
 Structure notes:
 ** Models **:
    This is the model called directly for training and testing. It calls:
   ** EncodeModel **:
      This takes in the input image set and outputs an encoded image set, split 
      into multiple subsets of features, with additional data used for a later 
      loss function. It calls:
     ** Split **:
        This takes in the input image batch and splits it into feature subsets
        for the encoders.
     ** Encode **:
        This takes in a subset of the input image features and outputs its 
        encoding.
   ** DecodeAndClassifyModel **:
      This takes in the encoded image set and outputs its classifications. It
      calls:
     ** Decode **:
        This recombines and transforms the encoded data to its original dimensions.
     ** Classify **:
        This takes in the recombined but still encoded image data
        and outputs its classifications.
Splitter, Encoder, Decoder, and Classifier each can be specified in manager.py.
'''

class Models(nn.Module):
    def __init__(self, subsets, dims, encoded_size, encoder_hidden_size, unencoded_dims, 
                 classes, classifier_hidden_size, splitter, encoder, decoder, 
                 classifier, classify_only, pretrained_classifier):
        super(Models, self).__init__()

        if (type(subsets[0]) == int):
            self.splitsubsets = subsets
        else:
            self.splitsubsets = []
            for i in range(len(subsets[0])):
                for j in subsets[1]:
                    self.splitsubsets.append(j)
            self.splitsubsets = tuple(self.splitsubsets)
        
        self.module = nn.ModuleDict({
                "encode": EncodeModel(subsets, self.splitsubsets, dims, unencoded_dims, encoded_size, 
                               encoder_hidden_size, splitter, encoder, classify_only), 
                "decode and classify": DecodeAndClassifyModel(subsets, self.splitsubsets, dims, 
                               unencoded_dims, encoded_size, classes, classifier_hidden_size, 
                               splitter, encoder, decoder, classifier, classify_only, pretrained_classifier)
                })
                
    def forward(self, x, module, *args, **kwargs):
        x = self.module[module](x, *args, **kwargs)
        return x

class EncodeModel(nn.Module):
    def __init__(self, subsets, splitsubsets, dims, unencoded_dims, encoded_size, encoder_hidden_size,
                 splitter, encoder, classify_only):
        super(EncodeModel, self).__init__()
        
        self.subsets = subsets
        self.splitsubsets = splitsubsets
        self.dims = dims
        
        self.splitter = splitter
        self.encoder = encoder
        self.unencoded_dims = unencoded_dims   
        self.classify_only = classify_only
        
        self.split = Split()
        
        if (not self.classify_only):
            self.encode = nn.ModuleList([])
            for subset in range(len(self.splitsubsets)):
                self.encode.append(Encode(self.splitsubsets[subset], encoder, encoded_size, encoder_hidden_size))
    
    def forward(self, x, outfile, classify_only, print_hooks = False, round_output = False):

        # Splitter reshapes data if needed, and then splits it. If only classifying,
        # reshaping is still needed
        if (self.classify_only):
            if (self.splitter == "1D"):
                x = x.reshape(-1, self.unencoded_dims['size'])
            return x, x
        else:
            xsplit = self.split(x, self.splitter, self.subsets, self.dims, self.unencoded_dims, self.classify_only)
            xround = []
            for subset in range(len(self.splitsubsets)):
                xsplit[subset] = self.encode[subset](xsplit[subset], self.encoder, outfile, print_hooks, round_output)
                with torch.no_grad():
                    xround.append(xsplit[subset].detach().clone().round())
                    
            return xsplit, xround
    
class DecodeAndClassifyModel(nn.Module):
    def __init__(self, subsets, splitsubsets, dims, unencoded_dims, encoded_size, classes,
                 classifier_hidden_size, splitter, encoder, decoder, classifier, classify_only, pretrained_classifier):
        super(DecodeAndClassifyModel, self).__init__()
        
        self.subsets = subsets
        self.splitsubsets = splitsubsets
        self.dims = dims
        if (encoder in ('AlexNetEncode', 'ResNetEncode', 'CustomAlexNetEncode')):
            self.unencoded_dims = { 'height': 32,
                      'width': 32,
                      'size': 32*32*3} # Resized RGB images
        else: 
            self.unencoded_dims = unencoded_dims
        self.encoded_size = encoded_size
        self.splitter = splitter
        self.decoder = decoder
        self.classify_only = classify_only
        
        if (not self.classify_only):
            self.decode = Decode(encoded_size*len(self.splitsubsets), self.unencoded_dims, self.encoded_size, self.decoder)
                
        self.classify = Classify(self.unencoded_dims, classes, classifier_hidden_size, classifier, pretrained_classifier)
    
    def forward(self, xsplit, outfile, classify_only, print_hooks = False):
        if(self.classify_only):
            x = xsplit
        else:
            # Decode it: unsplits and transforms data to original dimensions.
            # Dimensions are in shape that is useful for classifier.
            x = self.decode(xsplit, self.splitter, self.subsets, self.dims, outfile, print_hooks)

        del xsplit
        decodedsample = x.detach().clone()
        # Finally, classify
        x = self.classify(x)
        
        return x, decodedsample
        
class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()    
            
    def forward(self, x, splitter, subsets, dims, unencoded_dims, classify_only = False, revert = False):
        if (revert):
            if (splitter == '2D'):
                for i in range(len(x)):
                    x[i] = x[i].reshape(-1, 1, 1, x[i].size()[1])
                    
            return tools.unsplit_features(x, subsets, dims)
        else:
            # 1D and FC for RGB are used for linear encoders so flatten first.
            # Color Separation is used with a convolutional encoder but it's
            # convenient to implement as splitting between reshaping.
            if (splitter in ("1D", "FC for RGB", "Color Separation")):
                x = x.reshape(-1, unencoded_dims['size'])
            elif (splitter == "2D"):
                pass
            else: print ('Splitter not configured')
                
            x = tools.split_features(x, subsets, dims)
            
            if (splitter == "Color Separation"):
                for i in range(len(x)):
                    x[i] = x[i].reshape(-1, 1, unencoded_dims['height'], unencoded_dims['width'])
                
            return x

class Encode(nn.Module):
    def __init__(self, subset_size, encoder, encoded_size, encoder_hidden_size):
        super(Encode, self).__init__()
        
        if (encoder == "Simple 1D"):
            self.encoder = SimpleEncode(subset_size, encoded_size, encoder_hidden_size)
        elif (encoder == "Simple ReSigmoid"):
            self.encoder = SimpleEncodeReSigmoid(subset_size, encoded_size, encoder_hidden_size)
        elif (encoder == "Simple Quantize"):
            self.encoder = SimpleEncodeQuantize(subset_size, encoded_size, encoder_hidden_size)
        elif (encoder == "1 Channel Conv"):
            self.encoder = ConvEncode(1, encoded_size)
        elif (encoder == "3 Channel Small Conv"):
            self.encoder = ConvEncodeSmall(3, subset_size, encoded_size)
        elif (encoder == "1 Channel Conv B"):
            self.encoder = ConvEncodeB(1, encoded_size)
        elif (encoder == "1 Channel Conv C"):
            self.encoder = ConvEncodeC(1, encoded_size)
        elif (encoder == "1 Channel Conv Square"):
            self.encoder = ConvEncodeSquare(1, encoded_size)
        elif (encoder == "1 Channel Conv Square B"):
            self.encoder = ConvEncodeSquareB(1, encoded_size)
        elif (encoder == "1 Channel Conv Square 8x8"):
            self.encoder = ConvEncodeSquare8x8()
        elif (encoder == "AlexNetEncode"):
            self.encoder = AlexNetEncode(encoded_size)            
        elif (encoder == "CustomAlexNetEncode"):
            self.encoder = CustomAlexNetEncode(encoded_size)   
        elif (encoder == "ResNetEncode"):
            self.encoder = ResNetEncode(encoded_size)   
        else: print ('Encoder not configured')
            
    def forward(self, x, encoder, *args):
        x = self.encoder(x, *args)
        return x
    
class Decode(nn.Module):
    def __init__(self, encoded_input_size, unencoded_dims, encoded_size, decoder):
        super(Decode, self).__init__()
        
        self.unencoded_dims = unencoded_dims
        self.split = Split()
        if (decoder in ("Simple 1D")):
            self.dec = nn.Linear(encoded_input_size, unencoded_dims['size'])
        elif (decoder == "1 to 3 Channel Conv"):
            self.dec = ConvDecode1to3(unencoded_dims, encoded_size)
        elif (decoder == "3 Channel"):
            self.dec = ConvDecode3Channel(unencoded_dims, encoded_size)
        elif (decoder == "1 to 3 Channel Conv Square"):
            self.dec = ConvDecode1to3Square(unencoded_dims, encoded_size)
        elif (decoder == "1 to 3 Channel Conv Square B"):
            self.dec = ConvDecode1to3SquareB(unencoded_dims)
        elif (decoder == "1 to 3 Channel Conv Square 8x8"):
            self.dec = ConvDecode1to3Square8x8(unencoded_dims)
        else: print ('Decoder not configured')
    
    def forward(self, xsplit, splitter, subsets, dims, outfile, print_hooks):
        x = self.split(xsplit, splitter, subsets, dims, self.unencoded_dims, revert = True)
        x = self.dec(x)
        if(splitter == "FC for RGB"):
            x = x.reshape(-1, 3, self.unencoded_dims['height'], self.unencoded_dims['width'])
            
        if (print_hooks):
            x.register_hook(lambda a: print("3: x grad after decode: ", a[0][0:5], a.size(), file=outfile))  
        return x
    
class Classify(nn.Module):
    def __init__(self, unencoded_dims, classes, classifier_hidden_size, classifier, pretrained_classifier):
        super(Classify, self).__init__()

        if(pretrained_classifier):
            self.classifier = pretrained_classifier
            for param in self.classifier.parameters():
                param.requires_grad = False
        elif (classifier == "MNIST Tutorial"):
            self.classifier = SimpleClassify(unencoded_dims['size'], classifier_hidden_size, classes)
        elif (classifier == "CIFAR10 Tutorial"):
            self.classifier = ConvClassify(unencoded_dims, classes)
        elif (classifier == "WideResNet"):
            self.classifier =  WideResNet(10, classes, 1, 0)
        elif (classifier == "AlexNet"):
            self.classifier = models.AlexNet(classes)
        else: print ('Classifier not configured')
    
    def forward(self, x):
        x = self.classifier(x)
        return x

'''
### CONFIGURATION SPECIFIC MODULES ###
 These are used based on configurations in manager.py 
'''
    
class SimpleEncode(nn.Module):
    def __init__(self, subset_size, encoded_size, encoder_hidden_size):
        super(SimpleEncode, self).__init__()
        self.fc1 = nn.Linear(subset_size, encoder_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(encoder_hidden_size, encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()
            
    def forward(self, x, outfile, print_hooks, round_output):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if (print_hooks):
            x.register_hook(lambda a: print("1: x grad after encode fc2: ", a[0], a.size(), "\n", 
                                            "encode fc2 weights in backward: ", self.fc2.weight.data[0][0:5],
                                            self.fc2.weight.data.size(), file=outfile))
        x = self.norm(x)
        x = self.sig(x)
        if (print_hooks):
            x.register_hook(lambda a: print("2: x grad after encode sig: ", a[0], a.size(), file=outfile))
        if (round_output):
            x = x.round()
        return x
    
class SimpleEncodeReSigmoid(nn.Module):
    def __init__(self, subset_size, encoded_size, encoder_hidden_size):
        super(SimpleEncodeReSigmoid, self).__init__()
        self.fc1 = nn.Linear(subset_size, encoder_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(encoder_hidden_size, encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()
        self.resig = ReSigmoid()
            
    def forward(self, x, outfile, print_hooks, round_output):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.sig(x)
        x = self.resig(x)
        x = self.resig(x)
        x = self.resig(x)
        if (round_output):
            x = x.round()
        return x
    
class SimpleEncodeQuantize(nn.Module):
    def __init__(self, subset_size, encoded_size, encoder_hidden_size):
        super(SimpleEncodeQuantize, self).__init__()
        self.fc1 = nn.Linear(subset_size, encoder_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(encoder_hidden_size, encoded_size)
        self.sig = nn.Sigmoid()
        self.quantize = Quantize.apply
            
    def forward(self, x, outfile, print_hooks, round_output):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sig(x)  
        x = self.quantize(x)
        return x
    
class ConvEncode(nn.Module):
    '''
    Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''
    def __init__(self, channels, encoded_size):
        super(ConvEncode, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, outfile, print_hooks, round_output):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.norm(x)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        return x
    
class ConvEncodeB(nn.Module):
    def __init__(self, channels, encoded_size):
        super(ConvEncodeB, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Linear(16 * 5 * 5, encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, outfile, print_hooks, round_output):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc(x))
        x = self.norm(x)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        return x

class ConvEncodeC(nn.Module):
    def __init__(self, channels, encoded_size):
        super(ConvEncodeC, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 84)
        self.fc3 = nn.Linear(84, encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, outfile, print_hooks, round_output):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        x = self.norm(x)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        return x

class ConvEncodeSmall(nn.Module):
    def __init__(self, channels, subset_size, encoded_size):
        self.size = subset_size
        super(ConvEncodeSmall, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 3, padding = 1)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding = 1)
        self.fc1 = nn.Linear(16 * self.size * self.size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, outfile, print_hooks, round_output):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * self.size * self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.norm(x)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        return x

class ConvEncodeSquare(nn.Module):
    def __init__(self, channels, encoded_size):
        self.encoded_size = encoded_size
        super(ConvEncodeSquare, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 3, 3)
        self.conv3 = nn.Conv2d(3, 1, (7-self.encoded_size))
        self.norm = nn.BatchNorm1d(self.encoded_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, outfile, print_hooks, round_output):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.encoded_size, self.encoded_size)
        x = self.norm(x)
        x = x.view(-1, 1, self.encoded_size, self.encoded_size)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        return x
    
class ConvEncodeSquareB(nn.Module):
    def __init__(self, channels, encoded_size):
        self.encoded_size = encoded_size
        super(ConvEncodeSquareB, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 1, (7-self.encoded_size))
        self.norm = nn.BatchNorm1d(self.encoded_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, outfile, print_hooks, round_output):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.encoded_size, self.encoded_size)
        x = self.norm(x)
        x = x.view(-1, 1, self.encoded_size, self.encoded_size)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        return x
    
class ConvEncodeSquare8x8(nn.Module):
    def __init__(self):
        super(ConvEncodeSquare8x8, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 1, 3)
        self.norm = nn.BatchNorm1d(8)
        self.sig = nn.Sigmoid()

    def forward(self, x, outfile, print_hooks, round_output):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 8, 8)
        x = self.norm(x)
        x = x.view(-1, 1, 8, 8)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        return x
    
class AlexNetEncode(nn.Module):
    def __init__(self, encoded_size):
        super(AlexNetEncode, self).__init__()        
        self.conv = nn.Conv2d(1, 3, 5, padding = 2)
        self.classifier = models.alexnet(num_classes = encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, outfile, print_hooks, round_output):
        x = self.conv(x)
        x = self.classifier(x)
        x = self.norm(x)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        
        return x
    
class CustomAlexNetEncode(nn.Module):
    def __init__(self, encoded_size):
        super(CustomAlexNetEncode, self).__init__()        
        self.conv = nn.Conv2d(1, 3, 5, padding = 2)
        self.classifier = CustomAlexNet(encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, outfile, print_hooks, round_output):
        x = self.conv(x)
        x = self.classifier(x)
        x = self.norm(x)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        
        return x
    
class CustomAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
#            nn.Conv2d(192, 384, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(384, 256, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(256, 256, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(192 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
#            nn.Linear(4096, 4096),
#            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class ResNetEncode(nn.Module):
    def __init__(self, encoded_size):
        super(ResNetEncode, self).__init__()        
        self.conv = nn.Conv2d(1, 3, 5, padding = 2)
        self.classifier = models.resnet18(num_classes = encoded_size)
        self.norm = nn.BatchNorm1d(encoded_size)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, outfile, print_hooks, round_output):
        x = self.conv(x)
        x = self.classifier(x)
        x = self.norm(x)
        x = self.sig(x)
        if (round_output):
            x = x.round()
        
        return x

class ConvDecode1to3(nn.Module):
    '''
    Designed for splitting input into separate layers each for RGB
    '''
    def __init__(self, unencoded_dims, encoded_size):
        super(ConvDecode1to3, self).__init__()
        
        self.unencoded_dims = unencoded_dims
        
        self.fc1 = nn.Linear(encoded_size * 3, unencoded_dims['size'])

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(-1, 3, self.unencoded_dims['height'], self.unencoded_dims['width'])
        
        return x
    
class ConvDecode3Channel(nn.Module):
    '''
    Pairs with 3 Channel Small Conv (ConvEncodeSmall), for RGB input which has 
    been split out along dims 2 and 3. As is this has only been written to support
    the tested case of dividing the input in quarters, though it would take minimal
    effort to replace the hardcoded value of 4 with a value calculated using the
    total_subsets function in tools.
    '''
    def __init__(self, unencoded_dims, encoded_size):
        super(ConvDecode3Channel, self).__init__()
        
        self.unencoded_dims = unencoded_dims
        self.encoded_size = encoded_size
        
        self.fc1 = nn.Linear(encoded_size * 4, unencoded_dims['size'])

    def forward(self, x):
        x = x.reshape(-1, 1, 1, self.encoded_size*4)
        x = self.fc1(x)
        x = x.reshape(-1, 3, self.unencoded_dims['height'], self.unencoded_dims['width'])
        
        return x

class ConvDecode1to3Square(nn.Module):
    '''
    Designed only for 3x4x4 images (anticipates images have been rejoined as RGB
    after encoding to 4x4 size).
    '''
    def __init__(self, unencoded_dims, encoded_size):
        super(ConvDecode1to3Square, self).__init__()
        
        self.unencoded_dims = unencoded_dims
        
        self.upsample1 = nn.Upsample(scale_factor = 2, mode='bilinear') #4 to 8
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 2) # 8 to 10
        self.upsample2 = nn.Upsample(scale_factor = 2, mode='bilinear') #10 to 20
        self.conv2 = nn.Conv2d(6, 12, 5, padding = 3) # 20 to 22
        self.upsample3 = nn.Upsample(scale_factor = 2, mode='bilinear') #22 to 44
        self.conv3 = nn.Conv2d(12, 6, 7) # 44 to 38
        self.conv4 = nn.Conv2d(6, 3, 7) # 38 to 32
        
    def forward(self, x):
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
        x = self.upsample2(x)
        x = F.relu(self.conv2(x))
        x = self.upsample3(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x
    
class ConvDecode1to3SquareB(nn.Module):
    '''
    Designed only for 3x4x4 images (anticipates images have been rejoined as RGB
    after encoding to 4x4 size).
    '''
    def __init__(self, unencoded_dims):
        super(ConvDecode1to3SquareB, self).__init__()
        
        self.unencoded_dims = unencoded_dims
        
        self.upsample1 = nn.Upsample(scale_factor = 8, mode='bilinear') #4 to 32
        self.conv1 = nn.Conv2d(3, 3, 5, padding = 2)
        
    def forward(self, x):
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
        return x
    
class ConvDecode1to3Square8x8(nn.Module):
    '''
    Designed only for 3x8x8 images (anticipates images have been rejoined as RGB
    after encoding to 8x8 size).
    '''
    def __init__(self, unencoded_dims):
        super(ConvDecode1to3Square8x8, self).__init__()
        
        self.unencoded_dims = unencoded_dims
        
        self.upsample1 = nn.Upsample(scale_factor = 4, mode='bilinear') #8 to 32
        self.conv1 = nn.Conv2d(3, 6, 5, padding = 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = 2)
        self.conv3 = nn.Conv2d(16, 3, 5, padding = 2)
        
    def forward(self, x):
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class SimpleClassify(nn.Module):
    '''
    From https://medium.com/@athul929/hand-written-digit-classifier-in-pytorch-42a53e92b63e
    '''
    def __init__(self, unencoded_size, classifier_hidden_size, classes):
        super(SimpleClassify, self).__init__()
        self.layer1 = nn.Linear(unencoded_size, classifier_hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(classifier_hidden_size, classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class ConvClassify(nn.Module):
    '''
    From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    '''
    def __init__(self, unencoded_dims, classes):
        super(ConvClassify, self).__init__()
        self.unencoded_dims = unencoded_dims
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
BasicBlock, NetworkBlock, and WideResnet all from:
https://github.com/xternalz/WideResNet-pytorch
'''
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class LossFunction(nn.Module):
    def __init__(self, loss_function, mse_multiplier, subsets):
        super(LossFunction, self).__init__()
        
        self.mse_multiplier = mse_multiplier
        self.subsets = subsets
        self.loss_function = loss_function
        
        self.cel_lossfunction = nn.CrossEntropyLoss()
        
        if(loss_function == "CELWithEncodedMSE"):
            self.mse_lossfunction = MSELossMean(subsets)
            
    def forward(self, out, labels, *args):
        if(self.loss_function == "CELWithEncodedMSE"):
            mseloss = self.mse_lossfunction(self.subsets, *args)
            celloss = self.cel_lossfunction(out, labels)
            loss = mseloss + self.mse_multiplier * celloss
        else:
            loss = self.cel_lossfunction(out, labels)
            mseloss = None
            celloss = None
        
        return loss, mseloss, celloss

class MSELossMean(nn.Module):
    '''
    Loss function which takes a list of models, runs MSELoss on each, and then
    returns the mean of the result.
    '''
    def __init__(self, subsets):
        super(MSELossMean, self).__init__()
        
        self.MSE = nn.ModuleList([])
        for subset in range(tools.total_subsets(subsets)):
            self.MSE.append(nn.MSELoss())
    
    def forward(self, subsets, encoded, rounded, print_hooks, outfile):
        loss = 0
        for subset in range(len(encoded)):
            loss += self.MSE[subset](encoded[subset], rounded[subset])
            if (print_hooks):
                loss.register_hook(lambda a: print("5: loss MSE grad: ", a, file=outfile))              
        mean = loss / len(encoded)
        if (print_hooks):
            mean.register_hook(lambda a: print("5: loss MSE grad: ", a, file=outfile))  
        
        return mean

class ReSigmoid(nn.Module):
    '''
    This activation function is like the sigmoid function but is designed to be
    run on data that has already had sigmoid run on it, to make the output more
    closely approximate a step function. Each time it is run it will get closer
    to a unit step, but three times is nearly perfect to a couple decimal places.
    Custom Quantize function has better results and is shallower.
    '''
    def __init__(self):
        super(ReSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = 14*x - 7
        x = self.sigmoid(x)
        return x
    
class Quantize(torch.autograd.Function):
    '''
    This activation function rounds in the forward pass. As the gradient of the round
    function is zero, the backward pass instead uses the amount of rounding (ie, the 
    difference between rounded and unrounded values) as the gradient.
    '''
    def __init__(self):
        super(Quantize, self).__init__()
        
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        with torch.no_grad():
            output = input.round()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return torch.abs(input - input.round()) * grad_input

class StepApproximation(nn.Module):
    '''
    Takes a tensor of any dimension where each value in the tensor is to be rounded
    to a value between 0 and n, where n is the highest value which can be created with
    the provided number of bits (2**bits - 1). A quantization mapping is created in the
    form of a series of sigmoids. Values less than 0 are approximated to 0, and values
    greater than n are approximated to n. Assumes input in the 0-n range is already the
    desired and meaningful data; if all input is bounded between 0 and 1, and input bits
    value is 3, this will still effectively be a sigmoid.
    '''
    def __init__(self):
        super(StepApproximation, self).__init__()
        
        self.sig = nn.Sigmoid()        
    
    def forward(self, x, bits):
        s = torch.zeros_like(x)
        n = 2**bits
        for i in range (1, n):
            s+= self.sig(10*(x - i + 0.5))
        del x
        return s
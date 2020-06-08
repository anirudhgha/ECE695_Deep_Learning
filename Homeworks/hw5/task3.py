#!/usr/bin/env python

##  noisy_object_detection_and_localization.py

"""
MUCH OF THE CODE IN THIS FILE IS BASED ON PROF. AVINASH KAK'S DLSTUDIO CODE, 
SPECIFICALLY THE DETECTANDLOCALIZE CLASS. 

This task involved two parts. 
1. Building a cnn to recognize the noise content of an image
2. using the prediction from that cnn to redirect the flow of data into appropriately
trained secondary cnns which run the actual classification. 

Loadnet2 is used to detect the noise level of images, reaching 98% accuracy. 

The same loadnet2 is used to classify the images. There are 4 networks, each trained 
individually on each of the 4 noise level datasets. The 0% noise network and 20% noise
network are trained WITHOUT a gaussian smoothing layer, as that was found to decrease
accuracy in task 2. The 50% noise network and 80% noise network are trained WITH the 
gaussian smoothing layer in place. 

The accuracy for 0% and 20% networks don't change much from task2, but the higher 
noise networks fall in accuracy. I believe this may be because the noise detector
network is less accurate at differentiating between 50 and 80. 
"""

import random
import numpy
import torch
import os, sys
import torchvision
import torch.nn as nn
import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import pymsgbox
sys.path.insert(1, r'C:\Users\alasg\Documents\Course work\ECE695_DL\DLStudio-1.1.0\DLStudio')
from DLStudio import *

def foo():
    
    # BEGIN DEFINITION FOR DETECT AND LOCALIZE---------------------------------------------
    class DetectAndLocalize(nn.Module):             
        """
        The purpose of this inner class is to focus on object detection in images --- as
        opposed to image classification.  Most people would say that object detection
        is a more challenging problem than image classification because, in general,
        the former also requires localization.  The simplest interpretation of what
        is meant by localization is that the code that carries out object detection
        must also output a bounding-box rectangle for the object that was detected.

        You will find in this inner class some examples of LOADnet classes meant
        for solving the object detection and localization problem.  The acronym
        "LOAD" in "LOADnet" stands for

                    "LOcalization And Detection"

        The different network examples included here are LOADnet1, LOADnet2, and
        LOADnet3.  For now, only pay attention to LOADnet2 since that's the class I
        have worked with the most for the 1.0.7 distribution.
        """
        def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(DetectAndLocalize, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test
            self.dataset_counter = 0

        class PurdueShapes5Dataset(torch.utils.data.Dataset):
            def __init__(self, dl_studio, train_or_test, dataset_file, transform=None):
                super(DetectAndLocalize.PurdueShapes5Dataset, self).__init__()
                if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-20.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-50.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-80.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                elif train_or_test == 'train' and dataset_file == "load_all":
                    
#-------------------load in the 0% noise--------------------------
                    
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset_noise_0 = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                        self.label_map_noise_0 = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        
                        
                        # reverse the key-value pairs in the label dictionary:
#                        self.class_labels = dict(map(reversed, self.label_map_noise_0.items()))
#                        self.transform = transform

                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset_noise_0, self.label_map_noise_0 = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset_noise_0, self.label_map_noise_0 = pickle.loads(dataset)
                        torch.save(self.dataset_noise_0, "torch-saved-PurdueShapes5-10000-dataset.pt")
                        torch.save(self.label_map_noise_0, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
#                        self.class_labels = dict(map(reversed, self.label_map_noise_0.items()))
#                        self.transform = transform


    #-------------------load in the 20% noise--------------------------
    
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset_noise_20 = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                        self.label_map_noise_20 = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
#                        self.class_labels = dict(map(reversed, self.label_map_noise_20.items()))
#                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset_noise_20, self.label_map_noise_20 = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset_noise_20, self.label_map_noise_20 = pickle.loads(dataset)
                        torch.save(self.dataset_noise_20, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                        torch.save(self.label_map_noise_20, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
#                        self.class_labels = dict(map(reversed, self.label_map_noise_20.items()))
#                        self.transform = transform
                   
    #-------------------load in the 50% noise--------------------------
    
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset_noise_50 = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                        self.label_map_noise_50 = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
#                        self.class_labels = dict(map(reversed, self.label_map_noise_50.items()))
#                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset_noise_50, self.label_map_noise_50 = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset_noise_50, self.label_map_noise_50 = pickle.loads(dataset)
                        torch.save(self.dataset_noise_50, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                        torch.save(self.label_map_noise_50, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
#                        self.class_labels = dict(map(reversed, self.label_map_noise_50.items()))
#                        self.transform = transform
    
    #-------------------load in the 80% noise--------------------------
    
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset_noise_80 = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                        self.label_map_noise_80 = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
#                        self.class_labels = dict(map(reversed, self.label_map_noise_80.items()))
#                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset_noise_80, self.label_map_noise_80 = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset_noise_80, self.label_map_noise_80 = pickle.loads(dataset)
                        torch.save(self.dataset_noise_80, "torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                        torch.save(self.label_map_noise_80, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
#                        self.class_labels = dict(map(reversed, self.label_map_noise_80.items()))
  
    
    # all datasets have been loaded into their respective self.dataset's, now need to modify the labels of each self.dataset
                    idx = 0
                    while idx < len(self.dataset_noise_0):
                        self.dataset_noise_0[idx][4] = 0
                        self.dataset_noise_20[idx][4] = 1
                        self.dataset_noise_50[idx][4] = 2
                        self.dataset_noise_80[idx][4] = 3
                        idx += 1
    #create new dictionaries for 3 of the dicts, so that the keys (which are the # 0 to 10000) don't overlap
                    self.dataset = self.dataset_noise_0                
                    for i in range(10000, 10000 + len(self.dataset_noise_20)):
                        self.dataset.update({i: self.dataset_noise_20.get(i-10000)})
                    for i in range(20000, 20000 + len(self.dataset_noise_50)):
                        self.dataset.update({i: self.dataset_noise_50.get(i-20000)})
                    for i in range(30000, 30000 + len(self.dataset_noise_80)):
                        self.dataset.update({i: self.dataset_noise_80.get(i-30000)})
                        
    
    #create a new label map for the 0,20,50,80 dataset we want
                    self.label_map = {'noise 0%': 0, 'noise 20%': 1, 'noise 50%': 2, 'noise 80%': 3} 
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform    
    
    
   #-------------------DONE WITH LOADING THE TRAINING SETS AND MODIFYING/APPENDING THEM--------------------------



                elif train_or_test == 'test' and dataset_file == "load_all":
   #-------------------TESTING load in the 0% noise--------------------------
#load in the 0% noise testing set

                    dataset_file = "PurdueShapes5-1000-test.gz"
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset_noise_0, self.label_map_noise0 = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset_noise_0, self.label_map_noise0 = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
#                    self.class_labels = dict(map(reversed, self.label_map_noise0.items()))
#                    self.transform = transform
                        
#load in the 20% noise testing set
                   
                    dataset_file = "PurdueShapes5-1000-test-noise-20.gz"
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset_noise_20, self.label_map_noise20 = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset_noise_20, self.label_map_noise20 = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
#                    self.class_labels = dict(map(reversed, self.label_map_noise20.items()))
#                    self.transform = transform
                    
#load in the 50% noise testing set
                   
                    dataset_file = "PurdueShapes5-1000-test-noise-50.gz"
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset_noise_50, self.label_map_noise50 = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset_noise_50, self.label_map_noise50 = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
#                    self.class_labels = dict(map(reversed, self.label_map_noise50.items()))
#                    self.transform = transform
                    
#load in the 80% noise testing set
                    
                    dataset_file = "PurdueShapes5-1000-test-noise-80.gz"
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset_noise_80, self.label_map_noise80 = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset_noise_80, self.label_map_noise80 = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
#                    self.class_labels = dict(map(reversed, self.label_map_noise80.items()))
    
    # all datasets have been loaded into their respective self.dataset's, now need to modify the labels of each self.dataset
                    idx = 0
                    while idx < len(self.dataset_noise_0):
                        self.dataset_noise_0[idx][4] = 0
                        self.dataset_noise_20[idx][4] = 1
                        self.dataset_noise_50[idx][4] = 2
                        self.dataset_noise_80[idx][4] = 3
                        idx += 1
    #append all the datasets into one (they're all in self.dataset_noise_0)        
                    self.dataset = self.dataset_noise_0                
                    for i in range(1000, 1000 + len(self.dataset_noise_20)):
                        self.dataset.update({i: self.dataset_noise_20.get(i-1000)})
                    for i in range(2000, 2000 + len(self.dataset_noise_50)):
                        self.dataset.update({i: self.dataset_noise_50.get(i-2000)})
                    for i in range(3000, 3000 + len(self.dataset_noise_80)):
                        self.dataset.update({i: self.dataset_noise_80.get(i-3000)})                    
    
    #create a new label map for the 0,20,50,80 dataset we want
                    self.label_map = {'noise 0%': 0, 'noise 20%': 1, 'noise 50%': 2, 'noise 80%': 3} 
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform

    
 
                # all the INDIVIDUAL testing sets are loaded this way, but all_noise dataset will need to be appended
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
#             
            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
                im_tensor = torch.zeros(3,32,32, dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
                sample = {'image' : im_tensor, 
                          'bbox' : bb_tensor,
                          'label' : self.dataset[idx][4] }
                if self.transform:
                     sample = self.transform(sample)
                return sample

        def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       
#            transform = tvt.Compose([tvt.ToTensor(),
#                                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                               batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=0)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=0)
    
        class SkipBlock(nn.Module):
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DetectAndLocalize.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer = nn.BatchNorm2d
                self.bn = norm_layer(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

            def forward(self, x):
                identity = x                                     
                out = self.convo(x)                              
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)                              
                    out = self.bn(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class GaussianSmoothing(nn.Module):
            """
            CLASS DEFINED ON STACKOVERFLOW FOR A SIMPLE GAUSSIAN KERNEL THAT 
            SMOOTHES A 3X3 WINDOW TO GET A SINGLE PIXEL VALUE. 
            https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351
            
            This should help smooth some of the noise to increase the network's accuracy on noisy data.
            """
            def __init__(self, channels, kernel_size, sigma, dim=2):
                super(DetectAndLocalize.GaussianSmoothing, self).__init__()
                if isinstance(kernel_size, numbers.Number):
                    kernel_size = [kernel_size] * dim
                if isinstance(sigma, numbers.Number):
                    sigma = [sigma] * dim
        
                # The gaussian kernel is the product of the
                # gaussian function of each dimension.
                kernel = 1
                meshgrids = torch.meshgrid(
                    [
                        torch.arange(size, dtype=torch.float32)
                        for size in kernel_size
                    ]
                )
                for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                    mean = (size - 1) / 2
                    kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                              torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        
                # Make sure sum of values in gaussian kernel equals 1.
                kernel = kernel / torch.sum(kernel)
        
                # Reshape to depthwise convolutional weight
                kernel = kernel.view(1, 1, *kernel.size())
                kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
#                kernel = kernel.to(torch.device('cuda:0'))
#                self.weight = kernel
                self.register_buffer('weight', kernel)
                self.groups = channels
        
                if dim == 1:
                    self.conv = F.conv1d
                elif dim == 2:
                    self.conv = F.conv2d#.to(torch.device('cuda:0'))
                elif dim == 3:
                    self.conv = F.conv3d
                else:
                    raise RuntimeError(
                        'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
                    )
        
            def forward(self, input):
                return self.conv(input, weight=self.weight, groups=self.groups)

        class LOADnet2(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet2 uses both convo and linear layers for regression
            """ 
            def __init__(self, skip_connections=True, depth=32, smoothingon = False):
                super(DetectAndLocalize.LOADnet2, self).__init__()
                self.smoothingon = smoothingon
                self.pool_count = 3
                self.depth = depth // 2
                self.smoothing = DetectAndLocalize.GaussianSmoothing(channels=3, kernel_size=3, sigma=1)
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = DetectAndLocalize.SkipBlock(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64ds = DetectAndLocalize.SkipBlock(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DetectAndLocalize.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128 = DetectAndLocalize.SkipBlock(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128ds = DetectAndLocalize.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 5)
                ##  for regression
                self.conv_seqn = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.fc_seqn = nn.Sequential(
                    nn.Linear(16384, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)
                )
                
            def forward(self, x):
                # need to pad x before convolving with a gaussian kernel to smooth
                if self.smoothingon==True:
                    x = F.pad(x, (1, 1, 1, 1), mode='reflect')
                    x = self.smoothing(x)
                
                x = self.pool(torch.nn.functional.relu(self.conv(x)))
                ## The labeling section:
                x1 = x.clone()
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x1)                                               
                x1 = self.skip64ds(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x1)                                               
                x1 = self.skip64to128(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = self.skip128ds(x1)                                               
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                ## The Bounding Box regression:
                x2 = self.conv_seqn(x)
                x2 = self.conv_seqn(x2)
                # flatten
                x2 = x2.view(x.size(0), -1)
                x2 = self.fc_seqn(x2)
                return x1,x2

        class IOULoss(nn.Module):
            def __init__(self, batch_size):
                super(DetectAndLocalize.IOULoss, self).__init__()
                self.batch_size = batch_size
            def forward(self, input, target):
                composite_loss = []
                for idx in range(self.batch_size):
                    union = intersection = 0.0
                    for i in range(32):
                        for j in range(32):
                            inp = input[idx,i,j]
                            tap = target[idx,i,j]
                            if (inp == tap) and (inp==1):
                                intersection += 1
                                union += 1
                            elif (inp != tap) and ((inp==1) or (tap==1)):
                                union += 1
                    if union == 0.0:
                        raise Exception("something_wrong")
                    batch_sample_iou = intersection / float(union)
                    composite_loss.append(batch_sample_iou)
                total_iou_for_batch = sum(composite_loss) 
                return 1 - torch.tensor([total_iou_for_batch / self.batch_size])

        def run_code_for_training_with_CrossEntropy_and_BCE_Losses(self, net):        
            """
            BCE stands for the Binary Cross Entropy Loss that is used for
            the regression loss in this training method.
            """
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + "label.txt"
            filename_for_out2 = "performance_numbers_" + str(self.dl_studio.epochs) + "regres.txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion1 = nn.CrossEntropyLoss()
#            criterion2 = self.dl_studio.DetectAndLocalize.IOULoss(self.dl_studio.batch_size)
            criterion2 = nn.BCELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            for epoch in range(self.dl_studio.epochs):  
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                for i, data in enumerate(self.train_dataloader):
                    gt_too_small = False
                    inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                    if self.dl_studio.debug_train and i % 1000 == 999:
                        print("\n\n[iter=%d:] Ground Truth:     " % (i+1) + 
                        ' '.join('%5s' % self.dataserver_train.class_labels[labels[j].item()] for j in range(self.dl_studio.batch_size)))
                    inputs = inputs.to(self.dl_studio.device)
                    labels = labels.to(self.dl_studio.device)
                    bbox_gt = bbox_gt.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    if self.dl_studio.debug_train and i % 500 == 499:
                        inputs_copy = inputs.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>31] = 31
                        _, predicted = torch.max(outputs_label.data, 1)
                        print("[iter=%d:] Predicted Labels: " % (i+1) + 
                         ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()] 
                                           for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bbox_gt[idx][1])
                            i2 = int(bbox_gt[idx][3])
                            j1 = int(bbox_gt[idx][0])
                            j2 = int(bbox_gt[idx][2])
                            k1 = int(bbox_pc[idx][1])
                            k2 = int(bbox_pc[idx][3])
                            l1 = int(bbox_pc[idx][0])
                            l2 = int(bbox_pc[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            inputs_copy[idx,0,i1:i2,j1] = 255
                            inputs_copy[idx,0,i1:i2,j2] = 255
                            inputs_copy[idx,0,i1,j1:j2] = 255
                            inputs_copy[idx,0,i2,j1:j2] = 255
                            inputs_copy[idx,2,k1:k2,l1] = 255                      
                            inputs_copy[idx,2,k1:k2,l2] = 255
                            inputs_copy[idx,2,k1,l1:l2] = 255
                            inputs_copy[idx,2,k2,l1:l2] = 255
                        self.dl_studio.display_tensor_as_image(
                              torchvision.utils.make_grid(inputs_copy, normalize=True),
                             "see terminal for TRAINING results at iter=%d" % (i+1))
                    mask_regress = torch.zeros(self.dl_studio.batch_size,32,32,requires_grad=False)
                    mask_gt = torch.zeros(self.dl_studio.batch_size, 32,32)
                    for k,out_regres in enumerate(bbox_pred):
                        x1,y1,x2,y2 = bbox_pred[k].tolist()
                        x1_gt,y1_gt,x2_gt,y2_gt = bbox_gt[k].tolist()
                        x1,y1,x2,y2 = [int(item) if item >0 else 0 for item in (x1,y1,x2,y2)]
                        x1_gt,y1_gt,x2_gt,y2_gt = [int(item) if item>0 else 0 for item in (x1_gt,y1_gt,x2_gt,y2_gt)]
                        if abs(x1_gt - x2_gt)<5 or abs(y1_gt-y2_gt) < 5: gt_too_small = True
                        mask_regress_np = np.zeros((32,32), dtype=bool)
                        mask_gt_np = np.zeros((32,32), dtype=bool)
                        mask_regress_np[y1:y2,x1:x2] = 1
                        mask_gt_np[y1_gt:y2_gt, x1_gt:x2_gt] = 1
                        mask_regress[k,:,:] = torch.from_numpy(mask_regress_np)
                        mask_regress.reqiures_grad=True
                        mask_gt[k,:,:] = torch.from_numpy(mask_gt_np)
                        mask_gt.reqiures_grad=True                
                    loss_labeling = criterion1(outputs_label, labels)
                    loss_labeling.backward(retain_graph=True)        
                    loss_regression = criterion2(mask_regress, mask_gt)
                    loss_regression.requires_grad = True
                    loss_regression.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()                
                    if i % 1000 == 999:    
                        avg_loss_labeling = running_loss_labeling / float(1000)
                        avg_loss_regression = running_loss_regression / float(1000)
                        print("[epoch:%d, batch:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0
            print("\nFinished Training\n")
            self.save_model(net)

        def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + "label.txt"
            filename_for_out2 = "performance_numbers_" + str(self.dl_studio.epochs) + "regres.txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            for epoch in range(self.dl_studio.epochs):  
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                for i, data in enumerate(self.train_dataloader):
                    gt_too_small = False
                    inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                    if self.dl_studio.debug_train and i % 500 == 499:
#                    if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                        print("\n\n[epoch=%d iter=%d:] Ground Truth:     " % (epoch+1, i+1) + 
                        ' '.join('%10s' % self.dataserver_train.class_labels[labels[j].item()] for j in range(self.dl_studio.batch_size)))
                    inputs = inputs.to(self.dl_studio.device)
                    labels = labels.to(self.dl_studio.device)
                    bbox_gt = bbox_gt.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    if self.dl_studio.debug_train and i % 500 == 499:
#                  if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                        inputs_copy = inputs.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>31] = 31
                        bbox_pc[torch.isnan(bbox_pc)] = 0
                        _, predicted = torch.max(outputs_label.data, 1)
                        print("[epoch=%d iter=%d:] Predicted Labels: " % (epoch+1, i+1) + 
                         ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()] 
                                           for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bbox_gt[idx][1])
                            i2 = int(bbox_gt[idx][3])
                            j1 = int(bbox_gt[idx][0])
                            j2 = int(bbox_gt[idx][2])
                            k1 = int(bbox_pc[idx][1])
                            k2 = int(bbox_pc[idx][3])
                            l1 = int(bbox_pc[idx][0])
                            l2 = int(bbox_pc[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            inputs_copy[idx,0,i1:i2,j1] = 255
                            inputs_copy[idx,0,i1:i2,j2] = 255
                            inputs_copy[idx,0,i1,j1:j2] = 255
                            inputs_copy[idx,0,i2,j1:j2] = 255
                            inputs_copy[idx,2,k1:k2,l1] = 255                      
                            inputs_copy[idx,2,k1:k2,l2] = 255
                            inputs_copy[idx,2,k1,l1:l2] = 255
                            inputs_copy[idx,2,k2,l1:l2] = 255
#                        self.dl_studio.display_tensor_as_image(
#                              torchvision.utils.make_grid(inputs_copy, normalize=True),
#                             "see terminal for TRAINING results at iter=%d" % (i+1))
                    loss_labeling = criterion1(outputs_label, labels)
                    loss_labeling.backward(retain_graph=True)        
                    loss_regression = criterion2(bbox_pred, bbox_gt)
                    loss_regression.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()                
                    if i % 500 == 499:    
                        avg_loss_labeling = running_loss_labeling / float(500)
                        avg_loss_regression = running_loss_regression / float(500)
                        print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0
                    if self.dl_studio.debug_train and i%500==499:
#                    if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                        self.dl_studio.display_tensor_as_image(
                              torchvision.utils.make_grid(inputs_copy, normalize=True),
                             "see terminal for TRAINING results at iter=%d" % (i+1))


            print("\nFinished Training\n")
            self.save_model(net)

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing_detection_and_localization(self, net, net0, net20, net50, net80, mode, netnum):
            # writing output to output.txt file
            f = open('output.txt', 'a')
            
            #set datasetval so we can print correct value to output.txt
            if netnum == 0:
                datasetval = 'Noise'
            elif netnum == 1:
                datasetval = 'Dataset0'
            elif netnum == 2:
                datasetval = 'Dataset20'
            elif netnum == 3:
                datasetval = 'Dataset50'
            elif netnum == 4:
                datasetval = 'Dataset80'
            else:
                datasetval = ' '

            # set the batch size to 1 since every image needs to be rerouted
            self.dl_studio.batch_size = 1
            self.test_dataloader = torch.utils.data.DataLoader(self.dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=0)
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net0.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net20.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net50.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net80.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            
            correct = 0
            total = 0
            print(self.dataserver_train.class_labels)
            confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels), 
                                           len(self.dataserver_train.class_labels))
            class_correct = [0] * len(self.dataserver_train.class_labels)
            class_total = [0] * len(self.dataserver_train.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    images, bounding_box, labels = data['image'], data['bbox'], data['label']
                    labels = labels.tolist()
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
    self.dataserver_train.class_labels[labels[j]] for j in range(self.dl_studio.batch_size)))
                    outputs = net(images)
#                   if mode == 1, then this 'outputs' is what we want to find the error against
#                   if mode == 2, we want to use this output to run a second nn which will be what we calculate error against
                    if mode == 2:
                        outputs_label = outputs[0]
                        _, predicted = torch.max(outputs_label.data, 1)
                        if predicted == 0: # noise == 0%
                            # print('--USING NETWORK 0-NOISE')
                            outputs = net0(images)
                        elif predicted == 1: # noise == 20%
                            # print('--USING NETWORK 20-NOISE')
                            outputs = net20(images)
                        elif predicted == 2: # noise == 50%
                            # print('--USING NETWORK 50-NOISE')
                            outputs = net50(images)
                        elif predicted == 3: # noise == 80%
                            # print('--USING NETWORK 80-NOISE')
                            outputs = net80(images)


                    outputs_label = outputs[0]
                    outputs_regression = outputs[1]
                    outputs_regression[outputs_regression < 0] = 0
                    outputs_regression[outputs_regression > 31] = 31
                    outputs_regression[torch.isnan(outputs_regression)] = 0
                    output_bb = outputs_regression.tolist()
                    _, predicted = torch.max(outputs_label.data, 1)
                    predicted = predicted.tolist()
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
 self.dataserver_train.class_labels[predicted[j]] for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bounding_box[idx][1])
                            i2 = int(bounding_box[idx][3])
                            j1 = int(bounding_box[idx][0])
                            j2 = int(bounding_box[idx][2])
                            k1 = int(output_bb[idx][1])
                            k2 = int(output_bb[idx][3])
                            l1 = int(output_bb[idx][0])
                            l2 = int(output_bb[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            images[idx,0,i1:i2,j1] = 255
                            images[idx,0,i1:i2,j2] = 255
                            images[idx,0,i1,j1:j2] = 255
                            images[idx,0,i2,j1:j2] = 255
                            images[idx,2,k1:k2,l1] = 255                      
                            images[idx,2,k1:k2,l2] = 255
                            images[idx,2,k1,l1:l2] = 255
                            images[idx,2,k2,l1:l2] = 255
                        self.dl_studio.display_tensor_as_image(
                              torchvision.utils.make_grid(images, normalize=True), 
                              "see terminal for test results at i=%d" % i)
                    for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                    total += len(labels)
                    correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                    comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                    for j in range(self.dl_studio.batch_size):
                        label = labels[j]
                        class_correct[label] += comp[j]
                        class_total[label] += 1
            print("\n")
            for j in range(len(self.dataserver_train.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (
              self.dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
            print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % 
                                                                   (100 * correct / float(total)))
            # f.write(datasetval)
            f.write('{} Classification Accuracy: {}'.format(datasetval, 100*correct/float(total)))
            print("\n\nDisplaying the confusion matrix:\n")
            f.write('\n{} Confusion Matrix:\n'.format(datasetval))
            out_str = "                "
            for j in range(len(self.dataserver_train.class_labels)):  
                                 out_str +=  "%15s" % self.dataserver_train.class_labels[j]   
            print(out_str + "\n")
            f.write(out_str)
            f.write('\n\n')
            for i,label in enumerate(self.dataserver_train.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                 for j in range(len(self.dataserver_train.class_labels))]
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%12s:  " % self.dataserver_train.class_labels[i]
                for j in range(len(self.dataserver_train.class_labels)): 
                                                       out_str +=  "%15s" % out_percents[j]
                print(out_str)
                f.write(out_str)
                f.write('\n')
    
            f.close()
    # END DEFINITION FOR DETECT AND LOCALIZE---------------------------------------------
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    os.environ['PYTHONHASHSEED'] = str(seed)


    ##  watch -d -n 0.5 nvidia-smi

    dls_noise_level = DLStudio(
                      dataroot = "/content/drive/My Drive/Colab Notebooks/ece695_Deep_Learning/DLStudio-1.1.0/Examples/data/",
                      image_size = [32,32],
                      path_saved_model = "./saved_model_noise",
                      momentum = 0.9,
                      learning_rate = 1e-6,
                      epochs = 1,
                      batch_size = 4,
                      classes = ('noise 0%','noise 20%','noise 50%','noise 80%'),
                      debug_train = 1,
                      debug_test = 1,
                      use_gpu = True,
                  )


    dls_noise0 = DLStudio(
                      dataroot = "/content/drive/My Drive/Colab Notebooks/ece695_Deep_Learning/DLStudio-1.1.0/Examples/data/",
                      image_size = [32,32],
                      path_saved_model = "./saved_model_noise0",
                      momentum = 0.9,
                      learning_rate = 1e-6,
                      epochs = 2,
                      batch_size = 4,
                      classes = ('rectangle','triangle','disk','oval','star'),
                      debug_train = 1,
                      debug_test = 1,
                      use_gpu = True,
                  )
    dls_noise20 = DLStudio(
                      dataroot = "/content/drive/My Drive/Colab Notebooks/ece695_Deep_Learning/DLStudio-1.1.0/Examples/data/",
                      image_size = [32,32],
                      path_saved_model = "./saved_model_noise20",
                      momentum = 0.9,
                      learning_rate = 1e-6,
                      epochs = 2,
                      batch_size = 4,
                      classes = ('rectangle','triangle','disk','oval','star'),
                      debug_train = 1,
                      debug_test = 1,
                      use_gpu = True,
                  )
    dls_noise50 = DLStudio(
                      dataroot = "/content/drive/My Drive/Colab Notebooks/ece695_Deep_Learning/DLStudio-1.1.0/Examples/data/",
                      image_size = [32,32],
                      path_saved_model = "./saved_model_noise50",
                      momentum = 0.9,
                      learning_rate = 1e-6,
                      epochs = 2,
                      batch_size = 4,
                      classes = ('rectangle','triangle','disk','oval','star'),
                      debug_train = 1,
                      debug_test = 1,
                      use_gpu = True,
                  )
    dls_noise80 = DLStudio(
                      dataroot = "/content/drive/My Drive/Colab Notebooks/ece695_Deep_Learning/DLStudio-1.1.0/Examples/data/",
                      image_size = [32,32],
                      path_saved_model = "./saved_model_noise80",
                      momentum = 0.9,
                      learning_rate = 1e-6,
                      epochs = 2,
                      batch_size = 4,
                      classes = ('rectangle','triangle','disk','oval','star'),
                      debug_train = 1,
                      debug_test = 1,
                      use_gpu = True,
                  )
    

    #load all the noise datasets at once
    detector_noise_level = DetectAndLocalize(dl_studio = dls_noise_level)
    dataserver_train_allnoise = DetectAndLocalize.PurdueShapes5Dataset(
                                       train_or_test = 'train',
                                       dl_studio = dls_noise_level,
                                       dataset_file = "load_all",
                                                                          )
    dataserver_test_allnoise = DetectAndLocalize.PurdueShapes5Dataset(
                                        train_or_test = 'test',
                                        dl_studio = dls_noise_level,
                                        dataset_file = "load_all", 
                                      )
   # noisy = 0
    detector_noise0 = DetectAndLocalize( dl_studio = dls_noise0 )
    dataserver_train_0noise = DetectAndLocalize.PurdueShapes5Dataset(
                                    train_or_test = 'train',
                                    dl_studio = dls_noise0,
                                    dataset_file = "PurdueShapes5-10000-train.gz",
                                    )
    dataserver_test_0noise = DetectAndLocalize.PurdueShapes5Dataset(
                                    train_or_test = 'test',
                                    dl_studio = dls_noise0,
                                    dataset_file = "PurdueShapes5-1000-test.gz",
                              )
    # noisy = 20 
    detector_noise20 = DetectAndLocalize( dl_studio = dls_noise20 )
    dataserver_train_20noise = DetectAndLocalize.PurdueShapes5Dataset(
                                    train_or_test = 'train',
                                    dl_studio = dls_noise20,
                                    dataset_file = "PurdueShapes5-10000-train-noise-20.gz",
                                                                          )
    dataserver_test_20noise = DetectAndLocalize.PurdueShapes5Dataset(
                                    train_or_test = 'test',
                                    dl_studio = dls_noise20,
                                    dataset_file = "PurdueShapes5-1000-test-noise-20.gz" 
    )
    # noisy = 50
    detector_noise50 = DetectAndLocalize( dl_studio = dls_noise50 )
    dataserver_train_50noise = DetectAndLocalize.PurdueShapes5Dataset(
                                    train_or_test = 'train',
                                    dl_studio = dls_noise50,
                                    dataset_file = "PurdueShapes5-10000-train-noise-50.gz",
                                                                         )
    dataserver_test_50noise = DetectAndLocalize.PurdueShapes5Dataset(
                                    train_or_test = 'test',
                                    dl_studio = dls_noise50,
                                    dataset_file = "PurdueShapes5-1000-test-noise-50.gz"
    )

    # noisy = 80
    detector_noise80 = DetectAndLocalize( dl_studio = dls_noise80 )
    dataserver_train_80noise = DetectAndLocalize.PurdueShapes5Dataset(
                                    train_or_test = 'train',
                                    dl_studio = dls_noise80,
                                    dataset_file = "PurdueShapes5-10000-train-noise-80.gz",
                                                                          )
    dataserver_test_80noise = DetectAndLocalize.PurdueShapes5Dataset(
                                    train_or_test = 'test',
                                    dl_studio = dls_noise80,
                                    dataset_file = "PurdueShapes5-1000-test-noise-80.gz" 
                                        )

    f = open('output.txt', 'a')
    f.write('\n+Task 3:\n')
    f.close()
#   Train the noise level detector cnn
    
    detector_noise_level.dataserver_train = dataserver_train_allnoise
    detector_noise_level.dataserver_test = dataserver_test_allnoise
    detector_noise_level.load_PurdueShapes5_dataset(dataserver_train_allnoise, dataserver_test_allnoise) 
    model_noise_level = detector_noise_level.LOADnet2(skip_connections=True, depth=32, smoothingon=True)
    # dls.show_network_summary(model_noise_level)
    # detector_noise_level.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model_noise_level)


# #    # running model on noisy0
    detector_noise0.dataserver_train = dataserver_train_0noise
    detector_noise0.dataserver_test = dataserver_test_0noise
    detector_noise0.load_PurdueShapes5_dataset(dataserver_train_0noise, dataserver_test_0noise)
    model_noisy_0 = detector_noise0.LOADnet2(skip_connections=True, depth=32, smoothingon = False)
    dls_noise0.show_network_summary(model_noisy_0)
    # detector_noise0.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model_noisy_0)
#     #detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)
    
#     # running model on noisy20
    detector_noise20.dataserver_train = dataserver_train_20noise
    detector_noise20.dataserver_test = dataserver_test_20noise
    detector_noise20.load_PurdueShapes5_dataset(dataserver_train_20noise, dataserver_test_20noise)
    model_noisy_20 = detector_noise20.LOADnet2(skip_connections=True, depth=32, smoothingon = False)
    dls_noise20.show_network_summary(model_noisy_20)
    # detector_noise20.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model_noisy_20)
#     #detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)
    
#     # running model on noisy50
    detector_noise50.dataserver_train = dataserver_train_50noise
    detector_noise50.dataserver_test = dataserver_test_50noise
    detector_noise50.load_PurdueShapes5_dataset(dataserver_train_50noise, dataserver_test_50noise)
    model_noisy_50 = detector_noise50.LOADnet2(skip_connections=True, depth=32, smoothingon = True)
    dls_noise50.show_network_summary(model_noisy_50)
    # detector_noise50.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model_noisy_50)
#     #detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)
    
#     # running model on noisy80
    detector_noise80.dataserver_train = dataserver_train_80noise
    detector_noise80.dataserver_test = dataserver_test_80noise
    detector_noise80.load_PurdueShapes5_dataset(dataserver_train_80noise, dataserver_test_80noise)
    model_noisy_80 = detector_noise80.LOADnet2(skip_connections=True, depth=32, smoothingon = True)
    dls_noise80.show_network_summary(model_noisy_80)
    # detector_noise80.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model_noisy_80)
#     #detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)
# #
# #


#   first test the noisy model on how well it can detect noise
    detector_noise_level.dataserver_test = dataserver_test_allnoise
    detector_noise_level.run_code_for_testing_detection_and_localization(model_noise_level, model_noisy_0, model_noisy_20, model_noisy_50, model_noisy_80, mode=1, netnum=0)

#   need to run testing on each of the datasets individually
#   noise = 0
    detector_noise0.dataserver_test = dataserver_test_0noise
    detector_noise0.load_PurdueShapes5_dataset(dataserver_train_0noise, dataserver_test_0noise)
    detector_noise0.run_code_for_testing_detection_and_localization(model_noise_level, model_noisy_0, model_noisy_20, model_noisy_50, model_noisy_80, mode=2, netnum=1)

#   noise = 20
    detector_noise20.dataserver_test = dataserver_test_20noise
    detector_noise20.load_PurdueShapes5_dataset(dataserver_train_20noise, dataserver_test_20noise)
    detector_noise20.run_code_for_testing_detection_and_localization(model_noise_level, model_noisy_0, model_noisy_20, model_noisy_50, model_noisy_80, mode=2, netnum=2)

#   noise = 50
    detector_noise50.dataserver_test = dataserver_test_50noise
    detector_noise50.load_PurdueShapes5_dataset(dataserver_train_50noise, dataserver_test_50noise)
    detector_noise50.run_code_for_testing_detection_and_localization(model_noise_level, model_noisy_0, model_noisy_20, model_noisy_50, model_noisy_80, mode=2, netnum=3)

#   noise = 80
    detector_noise80.dataserver_test = dataserver_test_80noise
    detector_noise80.load_PurdueShapes5_dataset(dataserver_train_80noise, dataserver_test_80noise)
    detector_noise80.run_code_for_testing_detection_and_localization(model_noise_level, model_noisy_0, model_noisy_20, model_noisy_50, model_noisy_80, mode=2, netnum=4)








if __name__ == '__main__':
    foo()
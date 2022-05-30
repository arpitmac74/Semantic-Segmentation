import os, glob, pickle
import numpy as np
from utils import *

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision.models.segmentation
from unet.unet_model import UNet
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from evaluate import evaluate
import random

#
class SegmentationDataSet(Dataset):
    def __init__(self,inputs: list,targets: list,transform=bool):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)
    
    
    def __getitem__(self,
                    index: int):
        # Select the sample
        if self.targets is not None:
            x= self.inputs[index]
            y = self.targets[index]
        
            # Typecasting
            x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
            x ,y= x.permute(2,0,1),y.permute(2,0,1)
            #normalize the images
            x -= x.min(1,keepdim=True)[0]
            x /= x.max(1, keepdim=True)[0]
            if self.transform:
                if random.random() < 0.9:
                    x = self.transformation(x)
                return x, y
        else:
            x= self.inputs[index]
            x =torch.from_numpy(x).type(self.inputs_dtype)
            x = x.permute(2,0,1)

            #normalize the images
            x -= x.min(1,keepdim=True)[0]
            x /= x.max(1, keepdim=True)[0]
            return x
    
    @staticmethod
    def predict_preprocess(inputs,dtype = torch.float32):
        inputs = torch.from_numpy(inputs).type(dtype)
        inputs = inputs.squeeze(0)
        inputs = inputs.permute(2,0,1)
        inputs -= inputs.min(1,keepdim=True)[0]
        inputs /= inputs.max(1, keepdim=True)[0]
        return inputs
    
    #function to add augmentation in order to reduce domain difference
    @staticmethod
    def transformation(x):
        transform_1 = torchvision.transforms.Compose([
        torchvision.transforms.RandomInvert(p=1),
        torchvision.transforms.ColorJitter(contrast= 0)
    ])
        x = transform_1(x)
        return x
    

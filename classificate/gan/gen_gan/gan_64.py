import time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderConv(nn.Module):
    def __init__(self,noise_dim=100):
        super(EncoderConv, self).__init__()
        self.conv2dTransp0 = nn.ConvTranspose2d(in_channels=100,out_channels=8*64,kernel_size=4,stride=1,padding=0)
        self.batch_norm_t0=nn.BatchNorm2d(8*64)
        self.conv2dTransp1 = nn.ConvTranspose2d(in_channels=8*64,out_channels=4*64,kernel_size=4,stride=2,padding=1)
        self.batch_norm_t1=nn.BatchNorm2d(4*64)
        self.conv2dTransp2 = nn.ConvTranspose2d(in_channels=4*64,out_channels=2*64,kernel_size=4,stride=2,padding=1)
        self.batch_norm_t2=nn.BatchNorm2d(2*64)
        self.activ=nn.ReLU()
        self.conv2dTransp3 = nn.ConvTranspose2d(in_channels=2*64,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.batch_norm_t3=nn.BatchNorm2d(64)
        self.conv2dTransp4 = nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,stride=2,padding=1)
        self.tanh = nn.Tanh()
    def forward(self,x):
        x=x.view(-1,100,1,1)
        x=self.batch_norm_t0(self.activ(self.conv2dTransp0(x)))

        x=self.batch_norm_t1(self.activ(self.conv2dTransp1(x)))
        x=self.batch_norm_t2(self.activ(self.conv2dTransp2(x)))
        x=self.batch_norm_t3(self.activ(self.conv2dTransp3(x)))
        x=self.conv2dTransp4(x)
        x=self.tanh(x)
        return x.view(-1,3,64,64)
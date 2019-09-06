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
    def __init__(self,noise_dim=4096):
        super(EncoderConv, self).__init__()
        self.conv2dTransp1 = nn.ConvTranspose2d(in_channels=4096,out_channels=256,kernel_size=4,stride=2,padding=0)
        self.batch_norm_t1=nn.BatchNorm2d(256)
        self.conv2dTransp2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=4,padding=2)#,stride=2,padding=0)
        self.batch_norm_t2=nn.BatchNorm2d(256)
        self.conv2dTransp3 = nn.Conv2d(in_channels=256,out_channels=2*64,kernel_size=4,padding=1)#,stride=2,padding=0)
        self.batch_norm_t3=nn.BatchNorm2d(2*64)
        self.conv2dTransp4 = nn.Conv2d(in_channels=2*64,out_channels=64,kernel_size=3,padding=0)#,stride=1,padding=0)
        self.batch_norm_t4=nn.BatchNorm2d(64)
        self.conv2dTransp5 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=1)#,stride=2,padding=0)
        self.batch_norm_t5=nn.BatchNorm2d(32)
        self.conv2dTransp6 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,padding=1)#,stride=2,padding=0)
        self.batch_norm_t6=nn.BatchNorm2d(16)
        self.activ=nn.ReLU()
        self.upsample=nn.UpsamplingNearest2d(scale_factor=(2,2))
        self.conv2dTransp7 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,padding=1)#,stride=2,padding=0)
        self.batch_norm_t7=nn.BatchNorm2d(8)

        self.conv2dTransp8 = nn.Conv2d(in_channels=8,out_channels=3,kernel_size=3,padding=1)#,stride=2,padding=0)
        self.tanh = nn.Tanh()
    def forward(self,x):
        x=x.view(-1,4096,1,1)
        x=self.activ(self.conv2dTransp1(x))
        x=self.upsample(self.activ(self.batch_norm_t2(self.conv2dTransp2(x))))
        x=self.upsample(self.activ(self.batch_norm_t3(self.conv2dTransp3(x))))
        x=self.upsample(self.activ(self.batch_norm_t4(self.conv2dTransp4(x))))
        x=self.upsample(self.activ(self.batch_norm_t5(self.conv2dTransp5(x))))
        x=self.upsample(self.activ(self.batch_norm_t6(self.conv2dTransp6(x))))
        x=self.upsample(self.activ(self.batch_norm_t7(self.conv2dTransp7(x))))
        x=self.conv2dTransp8(x)
        x=self.tanh(x)
        return x.view(-1,3,256,256)
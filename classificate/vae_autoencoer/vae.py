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
from facenet_pytorch import InceptionResnetV1
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.extr =  nn.Sequential(*list(InceptionResnetV1(pretrained='vggface2').children())[:14])
        self.linear1 =nn.Linear(1792,1000)
        self.dropout=nn.Dropout(0.1)
        self.batch_norm1=nn.BatchNorm1d(1000)

        self.linear_mean=nn.Linear(1000,1000)
        self.batch_norm_mean=nn.BatchNorm1d(1000)
        self.linear_cov=nn.Linear(1000,1000)
        self.batch_norm_out=nn.BatchNorm1d(1000)

        self.linear_decoder2=nn.Linear(1000,512*15*15)
        self.batch_norm_d2=nn.BatchNorm1d(512*15*15)
        self.conv_t_0 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=0, stride=2)
        self.conv_b_0 = nn.BatchNorm2d(256)
        self.conv_t_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=3)
        self.conv_b_1 = nn.BatchNorm2d(128)
        self.conv_t_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv_b_2 = nn.BatchNorm2d(64)
        self.conv_t_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=3, stride=2)
        self.conv_b_3 = nn.BatchNorm2d(32)
        self.conv_t_4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=2, padding=2, stride=2)
        

        
    def forward(self, x):
        
        x = self.extr(x)
        x=x.view(-1,1792)

        x=self.batch_norm1(self.dropout(F.relu(self.linear1(x))))
        mean=self.batch_norm_mean(self.dropout(self.linear_mean(x)))
        cov = 0.5*self.linear_cov(x)
        x=self.batch_norm_out(mean+torch.exp(cov)*(torch.FloatTensor(1000).normal_().to(device)))
        return self.decoder(x),mean,cov
    def change_derivative(self,where):
        i=0
        for param in self.extr.parameters():
            i+=1
            if i>=where:
                param.requires_grad = True  
            else:
                param.requires_grad = False
    def decoder(self,x):
        x=self.batch_norm_d2(F.leaky_relu(self.linear_decoder2(x)))
        x=x.view(-1,512,15,15)
        x=self.conv_b_0(F.leaky_relu(self.conv_t_0(x)))
        x=self.conv_b_1(F.leaky_relu(self.conv_t_1(x)))
        x=self.conv_b_2(F.leaky_relu(self.conv_t_2(x)))
        x=self.conv_b_3(F.leaky_relu(self.conv_t_3(x)))
        x=self.conv_t_4(x)
        x = torch.tanh(x)
        return x
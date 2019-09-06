import time

#!pip install facenet_pytorch

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

        self.linear1 =nn.Linear(1792,100)
        self.dropout=nn.Dropout(0.3)
        self.batch_norm1=nn.BatchNorm1d(100)
        self.linear2 =nn.Linear(100,1)
    def forward(self, x):
        x=self.extr(x)
        x=x.view(-1,1792)
        x=self.batch_norm1(self.dropout(F.relu(self.linear1(x))))
        x=self.linear2(x)
        return x
    def change_derivative(self,where):
        i=0
        for param in self.extr.parameters():
            i+=1
            if i>=where:
                param.requires_grad = True  
            else:
                param.requires_grad = False
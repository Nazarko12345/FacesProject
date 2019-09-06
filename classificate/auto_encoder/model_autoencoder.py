import numpy as np
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

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64,64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128 )
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128 )
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv4d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn4d = nn.BatchNorm2d(256)
        
        self.conv3d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn3d = nn.BatchNorm2d(128)
        
        self.conv3dd = nn.Conv2d(128,  128, kernel_size=3, padding=1)
        self.bn3dd = nn.BatchNorm2d(128)
        self.dropout1=nn.Dropout2d(0.3)
        
        self.conv2d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(64)
        self.conv2dd = nn.Conv2d(64,64, kernel_size=3, padding=1)
        self.bn2dd = nn.BatchNorm2d(64)
        self.conv1d = nn.Conv2d(64, 3, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x1 = self.bn1(F.relu(self.conv1(x)))
        x1 = self.bn11(F.relu(self.conv11(x1)))

        x1p, id1 = F.max_pool2d(x1,kernel_size=2, stride=2,return_indices=True)
        # Stage 2
        x2 = self.bn2(F.relu(self.conv2(x1p)))
        x2 = self.bn22(F.relu(self.conv22(x2)))

        x2p, id2 = F.max_pool2d(x2,kernel_size=2, stride=2,return_indices=True)

        # Stage 3
        x3 = self.bn3(F.relu(self.conv3(x2p)))
        x3p, id3 = F.max_pool2d(x3,kernel_size=2, stride=2,return_indices=True)

        # Stage 4
        x4 = self.bn4(F.relu(self.dropout1(self.conv4(x3p))))
        #4d
        x4d = self.bn4d(F.relu(self.conv4d(x4)))

        # Stage 3d
        x3d = F.max_unpool2d(x4d, id3, kernel_size=2, stride=2,output_size=x3.size())
        x3d = self.bn3d(F.relu(self.conv3d(x3d)))
        x3d = self.bn3dd(F.relu(self.conv3dd(x3d)))

        # Stage 2d
        x2d = F.max_unpool2d(x3d, id2, kernel_size=2, stride=2,output_size=x2.size())
        x2d = self.bn2d(F.relu(self.conv2d(x2d)))
        x2d = self.bn2dd(F.relu(self.conv2dd(x2d)))

        # Stage 1d
        x1d = F.max_unpool2d(x2d, id1, kernel_size=2, stride=2,output_size=x1.size())
        x1d = F.tanh(self.conv1d(x1d))
        return x1d
    
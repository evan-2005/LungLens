import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CNNModel(nn.Module):
    def __init__(self, classCount, isTrained=True):
        super(CNNModel, self).__init__()
        # Encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Classification Head (Bottleneck features -> pool -> FC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, classCount),
            nn.Sigmoid()
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, classCount, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Classification Branch
        class_features = self.avgpool(x4)
        class_features = torch.flatten(class_features, 1)
        class_logits = self.classifier(class_features)
        
        # Decoder
        x_dec = self.up1(x4)
        x_dec = torch.cat([x_dec, x3], dim=1)
        x_dec = self.conv_up1(x_dec)
        
        x_dec = self.up2(x_dec)
        x_dec = torch.cat([x_dec, x2], dim=1)
        x_dec = self.conv_up2(x_dec)
        
        x_dec = self.up3(x_dec)
        x_dec = torch.cat([x_dec, x1], dim=1)
        x_dec = self.conv_up3(x_dec)
        
        logits = self.outc(x_dec)
        masks = self.sigmoid(logits)
        
        return class_logits, masks



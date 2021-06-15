import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from hparams import num_res
assert num_res > 1, 'Need to have atleast more than one res block (recommended minimum 3)'

# Basic Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, dim_in):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(dim_in, dim_in, kernel_size=3),
            nn.BatchNorm2d(dim_in), 
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in, dim_in, kernel_size=3),
            nn.BatchNorm2d(dim_in)) 
        
        self.final_activ = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.final_activ(out)
        return out
    

# Experimental (trying resbottlenecks instead of basic resblock)
# adapted from (https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py)
class ResidualBottleneck(nn.Module):
    def __init__(self, dim_in):
        super(ResidualBottleneck, self).__init__()
        
        expansion = 4
        dim_min = dim_in // expansion
        dim_out = dim_min * expansion
        
        stride = 1  # stride for downsampling
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_min, kernel_size=1),
            nn.BatchNorm2d(dim_min), 
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(dim_min, dim_min, kernel_size=3, stride=stride),
            nn.BatchNorm2d(dim_min), 
            nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim_min, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out))
        
        self.final_activ = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
#         print('=============')
#         print('input', x.shape)
        identity = x

        out = self.conv1(x)
#         print('res conv 1', out.shape)
        out = self.conv2(out)
#         print('res conv 2', out.shape)
        out = self.conv3(out)
#         print('res conv 3', out.shape)
#         print('=============')
        
        out += identity  # skip connection
        out = self.final_activ(out)
        return out



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Initial linear convolutional mapping
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        
        # Non-linear mapping convolutional layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv3 =nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))
             
        self.conv4 =nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))

        # Skip connections pre bottleneck
        res_modules = []
        for i in range(num_res):
            res_modules.append(ResidualBottleneck(1024))
        self.res5 = nn.Sequential(*res_modules)

        # Fully connected bottleneck
        self.fc6 = nn.Linear(1024, 512)
        self.mu7 = nn.Linear(512, 256)
        self.logvar7 = nn.Linear(512, 256)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):     
        
        # Main layers
        x = self.conv1(x)      
        x = self.conv2(x)      
        x = self.conv3(x)   
        x = self.conv4(x)
        
        # Residual phase before main bottleneck
        x = self.res5(x)
        
        # Bottleneck
        x = self.fc6(x.view(-1, 1024)) 
        mu = self.mu7(x)
        logvar = self.logvar7(x)   
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    
# Universal decoding residual block for first layer    
class ResGen(nn.Module):
    def __init__(self):
        super(ResGen, self).__init__()

        # Bottleneck opening
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        
        # Skip connections
        self.res1 = ResidualBottleneck(1024)
        
    def forward(self, x):
        x = self.fc1(x) 
        x = self.fc2(x) 
        x = self.res1(x.view(-1, 1024, 13, 13)) 
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # The first res block is shared so we do num_res - 1
        res_modules = []
        for i in range(num_res - 1):
            res_modules.append(ResidualBottleneck(1024))
        self.res1 = nn.Sequential(*res_modules)
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        
        # Final linear convolutional mapping 
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=11), 
            nn.Tanh())  # wrt DCGAN 
        
    def forward(self, x):
        x = self.res1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = self.conv4(x) 
        x = self.conv5(x) 
        return x


class Discriminator(nn.Module, ):
    def __init__(self, loss_mode='mse'):
        super(Discriminator, self).__init__()
        self.loss_mode = loss_mode
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(0.5) 
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(0.5) 
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(0.5) 
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Dropout(0.5) 
        )
                
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=8))
        self.linear_activ = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if (self.loss_mode == 'bce'): x = self.linear_activ(x) # only apply sigmoid like DCGAN when adv loss is BCE
        return x 
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim_in):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim_in, dim_in, kernel_size=4, bias=False),
            nn.InstanceNorm2d(dim_in),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in, dim_in, kernel_size=4, bias=False),
            nn.InstanceNorm2d(dim_in))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Initial linear convolutional mapping
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7, bias=False), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        
        # Non-linear mapping convolutional layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, bias=False, stride=2), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.conv3 =nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, bias=False, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        
        self.conv4 =nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, bias=False, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)) 
                 
        # Skip connection to bottleneck
        self.res5 = ResidualBlock(1024)

        # Fully connected bottleneck
        self.fc6 = nn.Linear(1024, 512)
        self.mu7 = nn.Linear(512, 256)
        self.logvar7 = nn.Linear(512, 256)
        
        # OLD: Starting with less residual blocks due to memory
        # l.append(ResidualBlock(512, 1024))
        # l.append(ResidualBlock(1024, 1156))
        # l.append(ResidualBlock(1156, 1280)) 
        
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
        x = self.res5(x) 
        print('res5', x.size())    
        
        # Bottleneck
        mu = self.mu7(x)
        print('mu7', x.size())  
        logvar = self.logvar7(x)   
        print('logvar7', x.size())  
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Bottleneck opening
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        
        # Skip connections
        self.res1 = ResidualBlock(1024)  
        
        # OLD: Starting with less residual blocks due to memory
        # l.append(ResidualBlock(1156, 1024))
        # l.append(ResidualBlock(1024, 512)) 
        
        # Non-linear mapping convolutional layers
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, bias=False, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, bias=False, stride=2), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, bias=False, stride=2), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        
        # Final linear convolutional mapping 
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=11, bias=False), 
            nn.Tanh())  # wrt DCGAN 
        
    def forward(self, x):
        x = self.fc1(x) 
        print('fc1', x.size())
        x = self.fc2(x) 
        print('fc2', x.size())
        x = self.res1(x)  
        print('res1', x.size())
        x = self.conv2(x) 
        print('conv2', x.size())
        x = self.conv3(x) 
        print('conv3', x.size())
        x = self.conv4(x) 
        print('conv4', x.size())
        x = self.conv5(x) 
        print('cov5', x.size())
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))
                
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=8, bias=False),
            nn.Sigmoid())  # wrt DCGAN

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x        

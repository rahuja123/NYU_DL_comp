import torch
from torch import nn
import torch.nn.functional as F

##Module so I don't have to keep rewriting the same conv
class conv_BN(nn.Module):
    def __init__(self, in_chanel, out_chanel):
        super(conv_BN, self).__init__()
        self.conv1 = nn.Sequential(
          nn.Conv2d(in_chanel,out_chanel,3,padding = 1),
          nn.BatchNorm2d(out_chanel), 
          nn.ReLU(),
        )
    def forward(self,input):
        output = self.conv1(input)
        return output

##Simple down block for autoencoder 
class conv_down(nn.Module):
    def __init__(self, in_chanel, out_chanel):
        super(conv_down, self).__init__()
        self.conv1 = conv_BN(in_chanel,out_chanel)
        self.conv2 = conv_BN(out_chanel,out_chanel)
        self.pooling1 = nn.MaxPool2d(2, stride=2)#8x128x128
    def forward(self,input):
        path1 = self.conv1(input)
        path1 = self.conv2(path1)
        path1 = self.pooling1(path1)
        return path1

##Simple up block for autoencoder
class conv_up(nn.Module):
    def __init__(self, in_chanel,out_chanel):
        super(conv_up, self).__init__()
        self.convUP = nn.Sequential(nn.ConvTranspose2d(in_chanel,out_chanel,2,stride = 2, padding = 0),
                                   nn.BatchNorm2d(out_chanel),
                                   nn.LeakyReLU())
        self.conv = conv_BN(out_chanel,out_chanel)
    def forward(self,input):
        path1 = self.convUP(input)
        path1 = self.conv(path1)
        return path1
    
class Encoder(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.conv1 = conv_down(3,64)
        self.conv2 = conv_down(64,128)
        self.conv3 = conv_down(128,256)
        self.conv4 = conv_down(256,512)
        self.conv5 = conv_down(512,512)
        hdim = (96//32)**2*512
        self.mu = nn.Linear(hdim, zdim)
        self.logvar = nn.Linear(hdim, zdim)
        self.zdim = zdim
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        h = self.conv5(out).view(x.shape[0],-1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu,logvar


class Decoder(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        hdim = (96//32)**2*512
        self.convt1 = conv_up(512,512)
        self.convt2 = conv_up(512,256)
        self.convt3 = conv_up(256,128)
        self.convt4 = conv_up(128,64)
        self.convt5 = conv_up(64,3)
        hdim = (96//32)**2*512
        self.linear = nn.Linear(zdim, hdim)

    def forward(self, x):
        out = self.linear(x).view(x.shape[0],512,3,3)
        out = self.convt1(out)
        out = self.convt2(out)
        out = self.convt3(out)
        out = self.convt4(out)
        out = self.convt5(out)
        out = F.hardtanh(out, min_val=0.0, max_val=1., inplace=False)
        return out

class VAE(nn.Module):
    def __init__(self,zdim):
        super().__init__()
        self.encoder = Encoder(zdim)
        self.decoder = Decoder(zdim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + std*eps
        else:
            z = mu
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)
        return xhat, mu, logvar
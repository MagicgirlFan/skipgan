""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import torch
import torch.nn as nn
import torch.nn.parallel
import functools
from torch.optim import lr_scheduler
from torch.nn import init
import numpy as np

class Conv_Block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
    def forward(self,x,bn=False):
        if bn:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.act(self.conv(x))

class Conv_DownSample(nn.Module):
    """
    Downsampling by convolution kernel 3*3 convolution kernel, step size 2，padding = 1
    """
    def __init__(self,in_ch,out_ch):
        super(Conv_DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
    def forward(self,x,bn=False):
        if bn:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.act(self.conv(x))

class MaxPool_DownSample(nn.Module):
    """
        Downsampling through pool 3*3 convolution kernel, step 2，padding = 1
    """
    def __init__(self,in_ch,out_ch,Max=True):
        super(MaxPool_DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        if Max:
            self.pool = nn.MaxPool2d(2,2)
        else:
            self.pool = nn.AvgPool2d(2,2)
    def forward(self,x,bn=True):
        if bn:
            return self.act(self.pool(self.bn(self.conv(x))))
        else:
            return self.act(self.pool(self.conv(x)))

class Conv_UpSample(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Conv_UpSample, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
    def forward(self,x,bn=False):
        if bn:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.act(self.conv(x))
class Net(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(Net, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,stride=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out
class Skpi_net(nn.Module):
    def __init__(self, in_channel=3, out_ch=64, n=5):
        super(Skpi_net, self).__init__()
        self.n = n
        in_ch = in_channel  # if n = 64
        self.Down_list = nn.ModuleList()
        self.Up_list = nn.ModuleList()
        for _ in range(n):
            self.Down_list.append(nn.Sequential(Conv_Block(in_ch, out_ch), MaxPool_DownSample(out_ch, out_ch)))
            in_ch = out_ch
            out_ch = out_ch * 2
        in_ch = 64 * (2 ** (n - 2))
        out_ch = 64 * (2 ** (n - 2))
        for i in range(n):
            if i == n - 1:
                out_ch = in_channel
                self.Up_list.append(nn.Sequential(Conv_UpSample(in_ch * 2, in_channel)))
            else:
                self.Up_list.append(nn.Sequential(Conv_UpSample(in_ch * 2, out_ch), Conv_Block(out_ch, out_ch)))
                in_ch = out_ch
                out_ch = in_ch // 2
        self.down_conv = nn.Sequential(*self.Down_list)
        self.up_conv = nn.Sequential(*self.Up_list)

    def forward(self, x):
        down_feature = []
        for i in range(self.n):
            x = self.down_conv[i](x)
            down_feature.append(x)
        for i in range(self.n):
            out = self.up_conv[i](x)
            if i < self.n - 1:
                x = torch.cat((down_feature[self.n - 2 - i], out), dim=1)
            else:
                x = out
        return x
class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.layer1 = MaxPool_DownSample(3,64)
        self.layer2 = MaxPool_DownSample(64, 128)
        self.layer3 = MaxPool_DownSample(128, 256)
        self.layer4 = MaxPool_DownSample(256, 512)
        self.layer5 = MaxPool_DownSample(512, 100)
        self.layer6 = nn.Linear(100*8*8,1)
        self.act = nn.Sigmoid()
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.flatten(1)
        x = self.act(self.layer6(x))
        return x
if __name__ == "__main__":
    net = Dis()
    data = torch.rand(2,6,256,256)
    out = net(data)
    print(out.shape)
    from torchsummary import summary

    summary(net,(6,256,256),device='cpu')
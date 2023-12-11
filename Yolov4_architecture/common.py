import torch.nn as nn
import torch
import torch.nn.functional as F
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
activate_name = {
    "relu": nn.ReLU(),
    "leaky": nn.LeakyReLU(inplace=True),
    'linear': nn.Identity(),
    "mish": nn.Mish()}
class conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,norm='True',activation='mish'):
            super(conv2d,self).__init__()
            self.norm=norm
            self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                            kernel_size=kernel_size,stride=stride,padding=kernel_size//2,bias=False)
            self.bn  =nn.BatchNorm2d(out_channels,eps=1e-3,momentum=0.03)
            self.activation=activate_name[activation]
    def forward(self,x):
            x=self.conv(x)
            if self.norm:
                x=self.bn(x)
            x=self.activation(x)
            return x
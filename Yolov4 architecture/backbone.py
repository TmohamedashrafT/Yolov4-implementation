from common import *

class CSPblock(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_channels='None'):
            super(CSPblock,self).__init__()
            if hidden_channels=='None':
                hidden_channels=out_channels ## for first stage
            self.block=nn.Sequential(
                conv2d(in_channels,hidden_channels,1),
                conv2d(hidden_channels,out_channels,3)
            )
    def forward(self,x):
            x= x+ self.block(x)
            return x
class FCSB(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FCSB,self).__init__()
        self.donwsample = conv2d(in_channels,out_channels, 3,stride=2)
        self.path_1     = conv2d(out_channels,out_channels,1)
        self.path_2     = conv2d(out_channels,out_channels,1)
        self.block      = nn.Sequential(
                          CSPblock(out_channels,out_channels,in_channels),
                          conv2d  (out_channels,out_channels,          1)
                            )
        self.lstconv    = conv2d(out_channels*2,out_channels,1)## out must be 64 
    def forward(self,x):
            x  =self.donwsample(x)
            x_1=self.path_1    (x)
            x_2=self.path_2    (x)
            x_2=self.block     (x_2)
            x  =torch.cat([x_2,x_1], dim=1)
            x  =self.lstconv(x)
            return x
class CSB(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks):
        super(CSB,self).__init__()
        self.downsample = conv2d(in_channels,out_channels,3,stride=2)
        self.path_1     = conv2d(out_channels,in_channels,1)
        self.path_2     = conv2d(out_channels,in_channels,1)
        
        self.block      = nn.Sequential(
                          *[CSPblock(in_channels,in_channels) for _ in range(num_blocks)],
                          conv2d(in_channels,in_channels,1)
                           )
        self.lstconv    = conv2d(out_channels,out_channels,1)
    def forward(self,x):
        x  =self.downsample(x)
        x_1=self.path_1(x)
        x_2=self.path_2(x)
        x_2=self.block(x_2)
        x  =torch.cat([x_2,x_1],dim=1)
        x  =self.lstconv(x)
        return x
class CSBDarknet53(nn.Module):
    def __init__(self,feature_channels=[64, 128, 256, 512, 1024],Routes=3):
        super(CSBDarknet53,self).__init__()
        self.fconv =conv2d(3,32,3)
        
        self.stages=nn.ModuleList([
            FCSB(32, feature_channels[0]),
            CSB(feature_channels[0], feature_channels[1], 2),
            CSB(feature_channels[1], feature_channels[2], 8),
            CSB(feature_channels[2], feature_channels[3], 8),
            CSB(feature_channels[3], feature_channels[4], 4)
            
        ])
        self.Routes=Routes
    def forward(self,x):
        x=self.fconv(x)
        all_routes=[]
        for stage in self.stages:
            x=stage(x)
            all_routes.append(x)
        return all_routes[-self.Routes:]
    
from common import *
class SPP(nn.Module):
    def __init__(self,feature_channels=[64, 128, 256, 512, 1024],pooling_size=[5,9,13],activation='leaky'):
        super(SPP, self).__init__()
        self.conv = nn.Sequential(
                    conv2d(feature_channels[-1],feature_channels[-2],1,activation=activation),
                    conv2d(feature_channels[-2],feature_channels[-1],3,activation=activation),
                    conv2d(feature_channels[-1],feature_channels[-2],1,activation=activation)
                    )
        self.pooling=nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pooling_size])
    def forward(self,x):
        x[2]=self.conv(x[2])
        features = [pooling(x[2]) for pooling in self.pooling]
        x[2] = torch.cat([x[2]]+features, dim=1)
        return x
class PaNet(nn.Module):
    def __init__(self,feature_channels=[64, 128, 256, 512, 1024],activation='leaky'):
        super(PaNet,self).__init__()
        '''torch.Size([1, 256, 56, 56])
        torch.Size([1, 512, 28, 28])
        torch.Size([1, 2048, 14, 14])'''
        self.conv_1=nn.Sequential(
                    conv2d(feature_channels[-1]*2,feature_channels[-2],1,activation=activation),
                    conv2d(feature_channels[-2],feature_channels[-1],3,activation=activation),
                    conv2d(feature_channels[-1],feature_channels[-2],1,activation=activation)
                        ) 
        self.conv_2=nn.Sequential(
                conv2d(feature_channels[-2],feature_channels[-3],1,activation=activation),
                nn.Upsample(scale_factor=2))
        self.conv_3=conv2d(feature_channels[-2],feature_channels[-3],1,activation=activation)
        ## concat 512
        self.conv_4=nn.Sequential(
                    conv2d(feature_channels[-2],feature_channels[-3],1,activation=activation),
                    conv2d(feature_channels[-3],feature_channels[-2],3,activation=activation),
                    conv2d(feature_channels[-2],feature_channels[-3],1,activation=activation),
                    conv2d(feature_channels[-3],feature_channels[-2],3,activation=activation),
                    conv2d(feature_channels[-2],feature_channels[-3],1,activation=activation)
                    )
        self.conv_5=nn.Sequential(
                conv2d(feature_channels[-3],feature_channels[-4],1,activation=activation),
                nn.Upsample(scale_factor=2))
        self.conv_6=conv2d(feature_channels[-3],feature_channels[-4],1,activation=activation)
        ## concat 256
        self.conv_7=nn.Sequential(
                    conv2d(feature_channels[-3],feature_channels[-4],1,activation=activation),
                    conv2d(feature_channels[-4],feature_channels[-3],3,activation=activation),
                    conv2d(feature_channels[-3],feature_channels[-4],1,activation=activation),
                    conv2d(feature_channels[-4],feature_channels[-3],3,activation=activation),
                    conv2d(feature_channels[-3],feature_channels[-4],1,activation=activation)
                    )     
        self.donwsample_8=conv2d(feature_channels[-4],feature_channels[-3],3,stride=2,activation=activation)
        ## concat 512
        self.conv_9=nn.Sequential(
                    conv2d(feature_channels[-2],feature_channels[-3],1,activation=activation),
                    conv2d(feature_channels[-3],feature_channels[-2],3,activation=activation),
                    conv2d(feature_channels[-2],feature_channels[-3],1,activation=activation),
                    conv2d(feature_channels[-3],feature_channels[-2],3,activation=activation),
                    conv2d(feature_channels[-2],feature_channels[-3],1,activation=activation)
                    ) 
        self.downsample_10=conv2d(feature_channels[-3],feature_channels[-2],3,stride=2,activation=activation)
        ## concat 1024
        self.conv_11=nn.Sequential(
                    conv2d(feature_channels[-1],feature_channels[-2],1,activation=activation),
                    conv2d(feature_channels[-2],feature_channels[-1],3,activation=activation),
                    conv2d(feature_channels[-1],feature_channels[-2],1,activation=activation),
                    conv2d(feature_channels[-2],feature_channels[-1],3,activation=activation),
                    conv2d(feature_channels[-1],feature_channels[-2],1,activation=activation)
                    )
        self.lstconvs = nn.ModuleList(
                    conv2d(feature_channels[i]//2, feature_channels[i], 3,activation='leaky') for i in range(2,5)
                      ) 
    def forward(self,x):
        x[2]=self.conv_1(x[2])
        x[1]=self.conv_3(x[1])
        x[0]=self.conv_6(x[0])
        w   =(self.conv_4(torch.cat([x[1],self.conv_2(x[2])],dim=1)))
        x[0]=self.conv_7(torch.cat([x[0],self.conv_5(w)],dim=1))
        x[1]=self.conv_9(torch.cat([self.donwsample_8(x[0]),w],dim=1))
        x[2]=self.conv_11(torch.cat([self.downsample_10(x[1]),x[2]],dim=1))
        x=[lstconv(feature_m) for lstconv, feature_m in zip(self.lstconvs, x)]
        return x

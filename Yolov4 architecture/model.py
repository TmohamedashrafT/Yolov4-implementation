from backbone import CSBDarknet53
from SPP_PaNet import *
from head import *
class Yolov4(nn.Module):
    def __init__(self,num_classes,stride,anchors=(),feature_channels=[64, 128, 256, 512, 1024]):
        super(Yolov4,self).__init__()
        self.model=nn.Sequential(CSBDarknet53(feature_channels),
        SPP(feature_channels),
        PaNet(feature_channels),
        head(anchors=anchors,stride=stride,num_classes=num_classes,lst_channels=feature_channels[-3:])
        )
    def forward(self,x):
        x=self.model(x)
        return   x
  
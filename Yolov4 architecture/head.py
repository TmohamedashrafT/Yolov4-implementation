import torch.nn as nn
import torch
class head(nn.Module):

    def __init__(self,
                 num_classes =1,
                 anchors = (),
                 stride  = [8,16,32],
                 lst_channels = [256, 512, 1024],
                 device='cuda'):  # detection layer
        super(head,self).__init__()
        self.num_classes = num_classes
        self.targets = num_classes + 5
        self.num_heads = len(anchors)
        self.num_anchors = len(anchors[0])
        self.device = device
        self.heads  = nn.ModuleList(nn.Conv2d(x, self.targets *  self.num_anchors, 1) for x in lst_channels)  # output conv
        self.stride = stride
        self.anchors= anchors.to(device= device )
                  
      
    def forward(self, x):
        z = []
        for i in range(self.num_heads):
            x[i] = self.heads[i](x[i])
            batch_size, _, grid_size, _ = x[i].shape # x shape : [batch_size, (num_classes +xywh + conf) * num_anchors, grid_size,grid_size]
            x[i] = x[i].view(batch_size, self.num_anchors, self.targets, grid_size, grid_size).permute(0, 3, 4, 1, 2)
            ## x shape :  [batch_size, grid_size, grid_size, num_anchors, (num_classes +xywh + conf)]
            xy, wh, conf = x[i][...,:2],x[i][...,2:4],x[i][...,4:]
            xy = (xy.sigmoid() + self.make_grid(grid_size, grid_size)) * self.stride[i]
            wh = (wh.exp() *  self.anchors[i]) * self.stride[i]
            y = torch.cat((xy, wh, conf.sigmoid()), 4)
            x[i][...,:] = y
            z.append( y.view(batch_size, self.num_anchors * grid_size * grid_size, self.targets))

        return x if self.training else torch.cat(z, 1)

    def make_grid(self, gridx, gridy):
        x  = torch.arange(gridx, device=self.device)
        y  = torch.arange(gridy, device=self.device)
        celly, cellx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((cellx, celly), 2)
        grid = grid.unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, 3, 1)
        return grid

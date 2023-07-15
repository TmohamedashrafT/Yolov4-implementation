
class head(nn.Module):
 
    def __init__(self, 
                 num_classes=1,
                 anchors=(), 
                 stride=[8,16,32],
                 lst_channels=[256, 512, 1024],
                 device='cuda'):  # detection layer
        super(head,self).__init__()
        self.num_classes = num_classes  
        self.targets = num_classes + 5 
        self.num_heads = len(anchors)  
        self.num_anchors = len(anchors[0])  
        self.grid = [torch.empty(0) for _ in range(self.num_heads)]  # init grid
        self.device=device
        self.heads = nn.ModuleList(nn.Conv2d(x, self.targets *  self.num_anchors, 1) for x in lst_channels)  # output conv
        self.stride=stride
        self.anchors=anchors.to(device='cuda')
        
    def forward(self, x):
        z = []
        for i in range(self.num_heads):
            x[i] = self.heads[i](x[i])  
            batch_size, _, gridy, gridx = x[i].shape  
            x[i] = x[i].view(batch_size, self.num_anchors, self.targets, gridy, gridx).permute(0, 3, 4, 1, 2).contiguous()

            if not self.training:  
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self.make_grid(gridx, gridy)
                xy, wh, conf = x[i][...,0:2],x[i][...,2:4],x[i][...,4:]
                xy = (xy.sigmoid() + self.grid[i]) * self.stride[i]  
                wh = (wh.exp()) *  (self.anchors[i] * self.stride[i]) 
                y = torch.cat((xy, wh, conf.sigmoid()), 4)
                z.append(y.view(batch_size, self.num_anchors * gridx * gridy, self.targets))
        
        return x if self.training else torch.cat(z, 1)

    def make_grid(self, gridx, gridy):
        x  = torch.arange(gridx,device=self.device) 
        y  = torch.arange(gridy,device=self.device)
        celly, cellx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((cellx, celly), 2)
        grid = grid.unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, 3, 1)
        return grid
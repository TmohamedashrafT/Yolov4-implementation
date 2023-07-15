from Config import cfg
from utils  import Early_stopping
from Compute_loss import ComputeLoss
from Dataset_Generator import data_loader
from torch.optim import lr_scheduler
import tqdm


class Training:
  def __init__(self,weight_path=None,device='cpu'):
    self.device=device
    self.best_map=0
    self.batch_size = cfg.model.batch_size
    self.model=Yolov4(cfg.model.num_classes,
             cfg.model.stride,
             cfg.model.anchors,
             cfg.model.feature_channels,
             ).to(device='cuda')
    self.optimizer = self.optimizer_( 
                                cfg.model.optimizer, 
                                cfg.model.lr , 
                                cfg.model.momentum, 
                                cfg.model.weight_decay)
    self.eatly_stopping= Early_stopping(patience=cfg.model.patience)
    if weight_path and weight_path[-3:]=='.pt':
      self.load_model(weight_path)
    self.compute_loss = ComputeLoss(device=cfg.model.device,
                                    num_classes=cfg.model.num_classes, 
                                    anchors    =cfg.model.anchors,
                                    balance    =cfg.model.head_scales,
                                    iou_thresh =cfg.model.iou_thresh ,
                                     )
    self.train_loader,_= data_loader(img_dir=cfg.model.train_image_path
                              ,ann_dir=cfg.model.train_ann_path
                              ,anchors=cfg.model.anchors,
                              iou_thresh=cfg.model.iou_wh_anchor,
                              batch_size=self.batch_size,
                              data_aug=cfg.model.data_aug.train)
    self.val_loader ,self.count_classes = data_loader(img_dir=cfg.model.val_image_path,
                                   ann_dir=cfg.model.val_ann_path,
                                   anchors=cfg.model.anchors,
                                   iou_thresh=cfg.model.iou_wh_anchor,
                                   batch_size=self.batch_size,
                                   train=False,
                                   data_aug=cfg.model.data_aug.val,
                                  
                                   )
    self.start_epoch = cfg.model.start_epoch
    self.epochs      = cfg.model.epochs
    self.accumulate  = 2
    self.classes     = dict(enumerate(cfg.model.classes)) 
    self.weights_path= cfg.model.weights_path
    self.img_size    = cfg.model.img_size
    self.freeze      = cfg.model.freeze
    lf = lambda x: (1 - x / 100) * (1.0 - cfg.model.lr) + cfg.model.lr
    self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)  
  def train(self):
    self.Freeze()
    #mloss = torch.zeros(3, device=self.device)
    print('################ Start  Training ################')
    for epoch in range(self.start_epoch,self.epochs):
      self.model.train()
      self.optimizer.zero_grad()

      for i,(imgs,targets,all_boxes) in enumerate(tqdm(self.train_loader)):
        imgs  = imgs.to(device=self.device,non_blocking=True).float()/255
        pred = self.model(imgs)
        loss,loss_item = self.compute_loss(pred,targets,all_boxes)  
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()  
        self.optimizer.zero_grad()

      print(f'epoch {epoch}/{self.epochs} box loss = ',loss_item[0].cpu().numpy(),'  conf loss = ',loss_item[1].cpu().numpy(),'  cls loss = ',loss_item[2].cpu().numpy(),'\n')
      map, maps  =eval_(device='cuda',
                            img_size=self.img_size,
                            model=self.model,
                            val_loader=self.val_loader,
                            classes=self.classes,
                            count_classes=self.count_classes)
      
      self.scheduler.step()
      self.save_model(epoch,map[0],map[1])
      ### early stopping 
      if self.eatly_stopping(self.weighted_map(map[0],map[1])):
        print('################ early_stopping ################')
        break
      ### end epoch 
    print('################ Training Finished ################')  


  def weighted_map(self,map50,map50_95,Wmap50=0.1,Wmap50_95=0.9):
      return map50 * Wmap50 + map50_95 * Wmap50_95


  def load_model(self,weights_path):
      ckpt = torch.load(weights_path, map_location=self.device)
      self.model.load_state_dict(ckpt['weights']) 
      self.optimizer.load_state_dict(ckpt['optimizer'])
      del ckpt

  def save_model(self, epoch,map50,map50_95):
      map_ = self.weighted_map(map50, map50_95)
      ckpt ={'weights':self.model.state_dict(),
            'mAP50':map50,
            'mAP50-95' :map50_95,
            'optimizer':self.optimizer.state_dict(),
            'epoch':epoch
            }
      torch.save(ckpt, self.weights_path+'last.pt')
      if map_>self.best_map:
          self.best_map=map_
          torch.save(ckpt, self.weights_path+'best.pt')
      del ckpt

  def Freeze(self,):
    freeze = [f'model.{x}.' for x in (self.freeze if len(self.freeze) > 1 else range(self.freeze[0]))]  # layers to freeze
    for k, v in self.model.named_parameters():
        v.requires_grad = True  
        if any(x in k for x in freeze): 
            v.requires_grad = False
  def optimizer_(self,opt_name='RMSProp',lr=0.001,momentum=0.9,weight_decay=1e-5):
      if opt_name=='SGD':
          optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
      elif opt_name=='Adam':
          optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(momentum,0.999),weight_decay=weight_decay)
      elif opt_name=='RMSProp':
          optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
      else:
          raise NotImplementedError(f'optimizer {opt_name} not implemented')
      return optimizer


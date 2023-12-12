from utils  import Early_stopping, read_yaml
from Compute_loss import ComputeLoss
from Dataset_Generator import data_loader
from torch.optim import lr_scheduler
from Yolov4_architecture.model import Yolov4 
from eval import eval_
from tqdm import tqdm
import torch
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from apex import amp
except:
    amp = None

class Training:
  def __init__(self, cfg, aug):
    self.device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.best_map = 0
    self.batch_size = cfg['batch_size']
    self.anchors = cfg['anchors']
    print('loading the model')
    self.model = Yolov4(num_classes = cfg['num_classes'],
                        anchors = cfg['anchors'],
                        device  = self.device,
                        ).to(device = self.device)
    self.optimizer = self.optimizer_(
                        cfg['optimizer'],
                        cfg['lr'] ,
                        cfg['momentum'],
                        cfg['weight_decay'])
    self.eatly_stopping= Early_stopping(patience = cfg['patience'])
    if  cfg['pretrained'] and  cfg['weights'][-3:]=='.pt':
      self.load_model(cfg['weights'])
    print('loading the data')
    self.train_loader,_= data_loader(img_dir =cfg['train_image_path'],
                                    ann_dir  = cfg['train_ann_path'],
                                    anchors  = cfg['anchors'],
                                    img_size = cfg['img_size'],
                                    data_aug = True,
                                    cfg_aug  = aug,
                                    max_boxes=cfg['max_boxes'],
                                    num_classes   = cfg['num_classes'],
                                    anchor_thresh = cfg['anchor_thresh'] , 
                                    batch_size = cfg['batch_size'],
                                    strides    = cfg['strides'],
                                    train = True
                              )
           
    self.val_loader ,self.count_classes = data_loader(img_dir    = cfg['val_image_path'],
                                                      ann_dir    = cfg['val_ann_path'],
                                                      anchors    = cfg['anchors'],
                                                      img_size   = cfg['img_size'],
                                                      num_classes= cfg['num_classes'],
                                                      batch_size = cfg['batch_size'],
                                                    )
    notest = False
    try:
        self.test_loader ,self.count_classes_test = data_loader(img_dir   = cfg['test_image_path'],
                                                      ann_dir    = cfg['test_ann_path'],
                                                      anchors    = cfg['anchors'],
                                                      img_size   = cfg['img_size'],
                                                      num_classes= cfg['num_classes'],
                                                      batch_size = cfg['batch_size'],
                                                    )
    except:
      notest = True
    self.epochs      = cfg['epochs']
    self.classes     = dict(enumerate(cfg['classes']))
    self.weights_path  = cfg['weights']
    self.img_size    = cfg['img_size']
    self.ComputeLoss = ComputeLoss(self.device, cfg['num_classes'],cfg['img_size'])
    self.fp16        = cfg['fp16']
    lf = lambda x: (1 - x / 100) * (1.0 - cfg['lr']) + cfg['lr']
    self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lf)  
  def train(self):
    print('################ Start  Training ################')
    if self.fp16 and amp:
        self.model, self.optimizer = amp.initialize(self.yolov4, self.optimizer, opt_level="O1")
    for epoch in range(self.epochs):
      self.model.train()
      box_loss = 0
      obj_loss = 0
      cls_loss = 0
      for i,(imgs, targets) in enumerate(tqdm(self.train_loader, ascii = True, desc ="Training")):
        imgs  = imgs.to(device=self.device, non_blocking=True).float() / 255
        preds = self.model(imgs)
        loss, loss_item =  self.ComputeLoss(preds, targets)
        box_loss   += loss_item[0].cpu().numpy()
        obj_loss   += loss_item[1].cpu().numpy()
        cls_loss   += loss_item[2].cpu().numpy()
        self.optimizer.zero_grad()
        if self.fp16 and amp:
             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
             loss.backward()
                
        self.optimizer.step()
        self.optimizer.zero_grad()
      self.scheduler.step()
      self.scheduler.get_last_lr()
      print(f'epoch {epoch}/{self.epochs} box loss = ',box_loss.round(decimals=3),'  conf loss = ',obj_loss.round(decimals=3),'  cls loss = ',cls_loss.round(decimals=3),'\n')
      map, maps  =eval_(device      = self.device,
                            batch_size = self.batch_size,
                            img_size   = self.img_size,
                            model      = self.model,
                            val_loader = self.val_loader,
                            classes    = self.classes,
                            count_classes = self.count_classes)
      
      self.save_model(epoch,map[0],map[1])
      ### early stopping
      if self.eatly_stopping(self.weighted_map(map[0],map[1])):
        print('################ early_stopping ################')
        break
      ### end epoch
    print('################ Training Finished ################')
    print('loading the best weights')
    self.load_model(os.path.join(self.weights_path, 'best.pt'))
    if not notest:
        self.val_loader = self.test_loader
        self.count_classes = self.count_classes_test
        
    map, maps  = eval_(device      = self.device,
                       batch_size  = self.batch_size,
                       img_size    = self.img_size,
                       model       = self.model,
                       val_loader  = self.val_loader,
                       classes     = self.classes,
                       count_classes = self.count_classes)           

  def weighted_map(self,map50,map50_95,Wmap50=0.1,Wmap50_95=0.9):
      return map50 * Wmap50 + map50_95 * Wmap50_95


  def load_model(self,weights_path):
      ckpt = torch.load(weights_path, map_location=self.device)
      self.model.load_state_dict(ckpt['weights'])
      self.optimizer.load_state_dict(ckpt['optimizer'])
      #self.best_map50_95=ckpt(['mAP50-95'])
      #self.best_map50   = ckpt(['mAP50'])
      self.best_map  = 0
      del ckpt

  def save_model(self, epoch,map50,map50_95):
      map_ = self.weighted_map(map50, map50_95)
      ckpt ={'weights':self.model.state_dict(),
            'mAP50':map50,
            'mAP50-95' :map50_95,
            'optimizer':self.optimizer.state_dict(),
            'epoch':epoch,
            'anchors':self.anchors,
             'classes': list(self.classes.values())
            }
      torch.save(ckpt, os.path.join(self.weights_path,'last.pt'))
      if map_>self.best_map:
          self.best_map=map_
          torch.save(ckpt, os.path.join(self.weights_path, 'best.pt'))
      del ckpt
  
  def optimizer_(self,opt_name='RMSProp',lr=0.001,momentum=0.9,weight_decay=1e-5):
      if   opt_name=='SGD':
          optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
      elif opt_name=='Adam':
          optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(momentum,0.999),weight_decay=weight_decay)
      elif opt_name=='RMSProp':
          optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
      else:
          raise NotImplementedError(f'optimizer {opt_name} not implemented')
      return optimizer





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, default = "data/BBCD.yaml", help = "data.yaml")  
    parser.add_argument("--hyp", type = str, default = "data/hyp.yaml", help = "hyp.yaml")  
    parser.add_argument("--aug", type = str,  default = "data/aug.yaml", help = "augmentation file")  
    parser.add_argument("--pretrained", type = bool, default = False, help = "Resume trainin")  
    parser.add_argument("--weights", type = str, default = "", help = "weights path")  
    parser.add_argument("--epochs", type = int, default = 100, help = "number of epochs")  
    parser.add_argument("--fp16", type = bool, default = False, help = "FP16 Training")  
    opt = parser.parse_args()
    data, hyp, aug =  read_yaml(opt.data), read_yaml(opt.hyp), read_yaml(opt.aug)
    cfg = dict()
    cfg.update(data)
    cfg.update(hyp)
    cfg['pretrained'] = opt.pretrained
    cfg['weights']    = opt.weights
    cfg['epochs']     = opt.epochs
    cfg['anchors']    = torch.tensor(cfg['anchors']).view(3,3,2)
    cfg['fp16']       = opt.fp16
    train = Training(cfg, aug)
    train.train()



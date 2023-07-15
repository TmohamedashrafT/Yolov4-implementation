from utils import load_image,xywh2xyxy,anchor_iou,xyxy2xywhn
from aug   import augment_hsv,albumentation

import numpy as np
import glob
import os
import torch
from torch.utils.data import DataLoader 

class LabelSmooth():
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes
def data_loader(img_dir,
                ann_dir,
                anchors,
                data_aug,
                batch_size= 16,
                shuffle   = True,
                train     = True,
                iou_thresh= 0.2,
                num_classes= 3
                ):
  dataset=Dataset_Generator(
          img_dir= img_dir,
          ann_dir= ann_dir,
          anchors= anchors,
          train  = train,
          iou_thresh = iou_thresh,
          data_aug   = data_aug,
          num_classes=num_classes
          )
  loader=DataLoader(
          dataset,
          batch_size=batch_size,
          shuffle=shuffle,
         )
  return loader,dataset.count_classes



class Dataset_Generator:
  def __init__(self,
                 img_dir,
                 ann_dir,
                 anchors     =(), 
                 img_size    =416,
                 data_aug    =True,
                 hyp         =None,
                 min_stride  =8,
                 max_boxes   =0,
                 train       =True,
                 num_classes =1,
                 iou_thresh  =0.2
               
                 ):
        self.img_size           = img_size
        self.data_aug           = data_aug
        self.min_stride         = min_stride
        self.max_boxes          = max_boxes
        self.img_files          = self.img_path(img_dir)
        self.num_classes        = num_classes
        self.count_classes      = np.zeros((self.num_classes))
        self.labels             = self.parse_annotation(ann_dir)
        self.label_smooth       = LabelSmooth()
        self.train              = train
        self.iou_thresh         = iou_thresh
        self.st                 = [1,2,4]
        self.anchors            = anchors
        self.number_of_anchors  = len(self.anchors)
  def __len__(self):
        return len(self.labels)
  def __getitem__(self, index):
        img, (oldh, oldw), (newh, neww), scale,pad    = load_image(self.img_files[index],self.img_size)

        labels                              = self.labels[index].copy()
        num_labels                          = len(labels)  
        ##labels (num_labels,class + x + y + w + h)

        ## rescale labels to new shape and returns to yolo format
        labels[:, 1:]     =  xywhn2xyxy(labels[:, 1:], scale *  oldw, scale * oldh, padw=pad[0], padh=pad[1])
        labels[:, 1:5]    =  xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        ## augmentation 
        if self.data_aug :
          augment_hsv(img)
          img,labels = albumentation(img,labels[...,1:],labels[...,0])


        labels            =  torch.tensor(labels)
        img = img.transpose((2, 0, 1))[::-1]  
        img = np.ascontiguousarray(img)

        #### for evaluation
        if not self.train:
          all_boxes       = torch.zeros((self.max_boxes, 5))
          all_boxes[:num_labels] = labels
          return torch.from_numpy(img), all_boxes


        grid_size  = [int(self.img_size/self.min_stride/i) for i in self.st]
        all_labels = [np.zeros((
                              grid_size[i],
                              grid_size[i],
                              self.number_of_anchors,
                              4+1+self.num_classes
                              ))for i in range(3)] ## (gridy,gridx,number of anchors,xywh+conf+number of class )
        all_boxes  = [np.zeros((self.max_boxes,4)) for _ in range(3)]

        for j in range(3):
          labels_with_gains = labels[...,1:]*grid_size[j]
          ious              = anchor_iou(labels_with_gains[...,2:4],self.anchors[j])  ## anchors shape(3,2) labels shape(nl,2) ious=(nl,3)
          ious_mask         = ious>self.iou_thresh
          intance_count     = 0  

          for i in range(num_labels):
              bbox_class_ind          = int(labels[i][0])
              one_hot                 = np.zeros(self.num_classes, dtype=np.float32)
              one_hot[bbox_class_ind] = 1.0
              one_hot_smooth          = self.label_smooth(one_hot, self.num_classes)
              gridx                   = (labels_with_gains[i][0]).long()
              gridy                   = (labels_with_gains[i][1]).long()
              x                       = (labels_with_gains[i][0]-gridx)
              y                       = (labels_with_gains[i][1]-gridy)

              box =  [x,y,labels_with_gains[i][2],labels_with_gains[i][3]]
              all_labels[j][ gridy, gridx, ious_mask[i],0:4]   = box
              all_labels[j][ gridy, gridx, ious_mask[i],4  ]   = 1
              all_labels[j][ gridy, gridx, ious_mask[i],5: ]   = one_hot_smooth
              all_boxes [j][intance_count]=box
              intance_count+=1
        return torch.from_numpy(img), all_labels,all_boxes
  def img_path(self,img_dir):
        img_files = []
        path      = Path(img_dir)
        img_files+=(glob.glob(str(path / '**' / '*.*'), recursive=True))
        return sorted(img_files)
  def parse_annotation(self,ann_dir):
        k=[]
        for label in sorted(os.listdir(ann_dir)):
          if label[-3:]!='txt' :
            continue
          with open(ann_dir+label) as t:
            t = t.read().strip().split() # class_id x_center y_center width height
            box=np.array([list(map(float, box.split(","))) for box in t],dtype=np.float32).reshape(int(len(t)/5),5)
            
            ind,ct=np.unique(box[...,0],return_counts=True)
            self.count_classes[ind.astype(int)]+=ct
            self.max_boxes=max(len(box),self.max_boxes)
            k.append(box)
        return k
  

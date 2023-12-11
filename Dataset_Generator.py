from utils import load_image, xywh2xyxy, xyxy2xywh, bb_iou
from aug   import augment_hsv, augs

import numpy as np
import glob
import os
import torch
from torch.utils.data import DataLoader 
from pathlib import Path

class LabelSmooth():
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes
def data_loader(img_dir,
                ann_dir,
                img_size,
                num_classes,
                batch_size,
                cfg_aug = None,
                anchors  = (),
                strides  = [8,16,32],
                max_boxes = 50,  
                train = False,
                data_aug = False,
                anchor_thresh = 0.3,
                shuffle = True,
                num_workers = 3,
                ):
    
  
  dataset = Dataset_Generator(
          img_dir     = img_dir,
          ann_dir     = ann_dir,
          anchors     = anchors,
          img_size    = img_size,
          data_aug    = data_aug,
          strides     = strides,
          max_boxes   = max_boxes,
          train       = train,
          num_classes = num_classes,
          anchor_thresh = anchor_thresh,
          cfg_aug   = cfg_aug
          )
  loader = DataLoader(
          dataset,
          batch_size = batch_size,
          shuffle = shuffle,
          num_workers = num_workers
         )
  return loader, dataset.count_classes


class Dataset_Generator:
  def __init__(self,
                 img_dir,
                 ann_dir, 
                 anchors,    
                 img_size,    
                 data_aug,   
                 strides,     
                 max_boxes,   
                 train,      
                 num_classes, 
                 anchor_thresh,
                 cfg_aug,
                 ):
    
        
        self.img_size           = img_size
        self.data_aug           = data_aug
        self.max_boxes          = max_boxes
        self.img_files          = self.img_path(img_dir)
        self.num_classes        = num_classes
        self.count_classes      = np.zeros((num_classes))
        self.labels             = self.parse_annotation(ann_dir)
        self.label_smooth       = LabelSmooth()
        self.train              = train
        self.anchor_thresh      = anchor_thresh
        self.anchors            = anchors
        self.number_of_anchors  = len(self.anchors)
        self.strides            = torch.tensor(strides)
        self.cfg_aug            = cfg_aug
  def __len__(self):
        return len(self.img_files)
  def __getitem__(self, index):
        img, (oldh, oldw), (newh, neww), scale,pad  = load_image(self.img_files[index],self.img_size)
        shapes                              = (oldh, oldw) ,(scale, pad)

        labels                              = self.labels[index].copy()
        num_labels                          = len(labels)
        ##labels (num_labels,class + x + y + w + h)
        ## rescale labels to new shape
        labels[:, 1:]     =  xywh2xyxy(labels[:, 1:], scale *  oldw, scale * oldh, padw=pad[0], padh=pad[1])
        labels[:, 1:5]    =  xyxy2xywh(labels[:, 1:5], w=img.shape[1], h=img.shape[0])
        ## augmentation
       
        if self.data_aug and self.cfg_aug is not None:
          augment_hsv(img)
          img,labels = augs(img,labels[...,1:],labels[...,0],self.cfg_aug)

        labels =  torch.tensor(labels)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        #### for evaluation
        if not self.train:
          all_boxes   = torch.zeros((self.max_boxes, 5))
          all_boxes[:num_labels] = labels
          return torch.from_numpy(img), all_boxes, shapes

        
        grid_size  = [int(self.img_size/stride) for stride in self.strides]
        all_labels = [np.zeros((
                              grid_size[i],
                              grid_size[i],
                              self.number_of_anchors,
                              4 + 1 + self.num_classes
                              ))for i in range(3)] ## (gridy,gridx,number of anchors,xywh+conf+number of classes )
        for label in labels:
          one_hot   = torch.zeros(self.num_classes)
          class_ind = int(label[0])
          one_hot[class_ind]= 1.0
          one_hot_smooth    = self.label_smooth(one_hot, self.num_classes)
          labels_with_gains = (label[1:] * self.img_size).type(torch.int64)
          bbox_xywh_scaled  = 1.0 * labels_with_gains[np.newaxis, :] / self.strides[:, np.newaxis]
          iou = []
          exist_positive = False
          for j in range(self.number_of_anchors):
            anchors_xywh = torch.zeros((3, 4))
            anchors_xywh[:, 0:2] = (torch.floor(bbox_xywh_scaled[j, 0:2])).type(torch.int64) + 0.5
            anchors_xywh[:, 2:4] =  self.anchors[j]
            iou_scale = bb_iou(bbox_xywh_scaled[j][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > self.anchor_thresh

            if torch.any(iou_mask):
                gridx, gridy = torch.floor(bbox_xywh_scaled[j, 0:2]).type(torch.int64)
                all_labels[j][gridy, gridx, iou_mask, 0:4] = labels_with_gains
                all_labels[j][gridy, gridx, iou_mask, 4:5] = 1.0
                all_labels[j][gridy, gridx, iou_mask, 5:] = one_hot_smooth
                exist_positive = True
          if not exist_positive:
                best_anchor_ind = torch.argmax(torch.cat(iou,0))
                best_scale  = int(best_anchor_ind / 3)
                best_anchor = int(best_anchor_ind % 3)
                gridx, gridy = torch.floor(bbox_xywh_scaled[best_scale, 0:2]).type(torch.int64)
                all_labels[best_scale][gridy, gridx, best_anchor, 0:4] = labels_with_gains
                all_labels[best_scale][gridy, gridx, best_anchor, 4:5] = 1.0
                all_labels[best_scale][gridy, gridx, best_anchor, 5:]  = one_hot_smooth
        return torch.from_numpy(img), all_labels
  def img_path(self,img_dir):
        img_files = []
        path      = Path(img_dir)
        img_files+=(glob.glob(str(path / '**' / '*.*'), recursive=True))
        return sorted(img_files)
  def parse_annotation(self,ann_dir):
        labels = []
        for ind,label in enumerate(sorted(os.listdir(ann_dir))):
          if label[-3:] != 'txt' or label == 'classes.txt' :
            continue
          label_path = os.path.join(ann_dir, label)
          assert label_path.split('/')[-1].split('.')[0] == self.img_files[ind].split('/')[-1].split('.')[0],\
            f'this path {label_path} has no image '
          with open(label_path) as t:
              t = t.read().strip().split() # class_id x_center y_center width height
              box = np.array([list(map(float, box.split(','))) for box in t],dtype=np.float32).reshape(int(len(t)/5),5)

              ind,ct = np.unique(box[...,0],return_counts=True)
              self.count_classes[ind.astype(int)] += ct
              self.max_boxes = max(len(box),self.max_boxes)
              labels.append(box)

        return labels

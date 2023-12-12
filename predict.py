from utils import NMS, load_image, draw_boxes
import torch
import os
import numpy as np
import cv2
from Yolov4_architecture.model import Yolov4    
import time
import argparse
def pred(img_path = '',
         conf_thresh = 0.1,
         iou_thresh  = 0.45,
         classes = [],
         device  = 'cuda',
         weights = None,
         img_size= 224):
  if weights:
    ckpt  = torch.load(weights,map_location='cpu')
    model = Yolov4(num_classes = len(ckpt['classes']),
                   anchors = ckpt['anchors'],
                    device = device,).to(device =device)
    model.load_state_dict(ckpt['weights'])
    classes = ckpt['classes'] 
    del ckpt 
  img, (oldh, oldw), (newh,neww),scale,(dw,dh)=load_image(img_path,img_size)
  shapes  = scale, (dw,dh)
  img = img.transpose((2, 0, 1))[::-1]
  img = np.ascontiguousarray(img)
  img = torch.from_numpy(img[None])
  img = img.to(device, non_blocking=True).float()/255
  t0 = time.time()
  model.eval()
  with torch.no_grad():
     preds=model(img)
  t1 = time.time()
  preds = NMS(preds,conf_thresh,iou_thresh)
  t2 = time.time()
  print('prediction speed = ',t1 - t0,'NMS speed = ',t2 - t1) 
  draw_boxes(img_path,preds,shapes,classes)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type = str, default = "weights/best.pt", help = "weights path")  
    parser.add_argument("--src", type = str, default = "img_test/img1.jpg", help = "source")  
    parser.add_argument("--conf_thresh", type = float,  default = 0.01, help = "confidence threshold")  
    parser.add_argument("--iou_thresh", type = float, default = 0.45, help = "NMS threshold")  
    parser.add_argument("--img_size", type = int, default = 224, help = "image size")  
    opt = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred(img_path = opt.src,
         conf_thresh = opt.conf_thresh,
         iou_thresh  = opt.iou_thresh,
         device  = device,
         weights = opt.weights,
         img_size= opt.img_size)

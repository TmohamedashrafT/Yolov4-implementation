import torch 
import numpy as np
import cv2
import math
import torchvision
import yaml
import random
def read_yaml(path):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
def xywh2xyxy(x, w = 1, h = 1, padw = 0, padh = 0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y
def xyxy2xywh(x, w, h):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

def load_image(img_path,img_size):
        img      = cv2.imread(img_path)
        assert img is not None,'{img_path} not founded'
        h,w,_ = img.shape
        ih,iw = img_size,img_size
        scale = min(ih/h,iw/w)
        scale = min(scale, 1.0)
        th,tw = int(scale*h),int(scale*w)
        if scale != 1:  # if sizes are not equal
          img=cv2.resize(img,(tw,th))
        dw, dh      = (iw - tw) / 2, (ih-th) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img         = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
        return img, (h, w), img.shape[:2],scale,(dw,dh)  # im, hw_original, hw_resized


class Early_stopping:
    def __init__(self, patience=30):
        self.best_map = 0.0 
        self.fipatience = patience
        self.patience = patience

    def __call__(self, mAP):
        if mAP >= self.best_map: 
            self.patience = self.fipatience
            self.best_map = mAP
        else :
            self.patience -= 1
        stop = True if not self.patience else False
        return stop


def bb_iou(boxes1,boxes2,fun='iou',format='xywh'):
    '''
    boxes format xywh
    boxes shape (batch_size,gridy,gridx,num_anchors,4)


    '''
    # xywh->xyxy
    if format=='xywh':
      boxes1=xywh2xyxy(boxes1, w=1, h=1, padw=0, padh=0)
      boxes2=xywh2xyxy(boxes2, w=1, h=1, padw=0, padh=0)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_corner = torch.max(boxes1[...,0:2],boxes2[...,0:2])
    right_corner= torch.min(boxes1[...,2:4],boxes2[...,2:4])

    inter       = torch.max(right_corner-left_corner,torch.zeros_like(right_corner))
    inter       = inter[...,0]*inter[...,1]
    union       = boxes1_area+boxes2_area-inter
    iou  = inter/ ( union + 1e-7 )
    if fun=='iou':
      return iou

    # cal outer boxes
    outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(right_corner))
    outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2)

    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
    center_dis = torch.pow(boxes1_center[...,0]-boxes2_center[...,0], 2) +\
                 torch.pow(boxes1_center[...,1]-boxes2_center[...,1], 2)

    boxes1_size = torch.max(boxes1[..., 2:] - boxes1[..., :2], torch.zeros_like(right_corner))
    boxes2_size = torch.max(boxes2[..., 2:] - boxes2[..., :2], torch.zeros_like(right_corner))
    v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan((boxes1_size[...,0]/torch.clamp(boxes1_size[...,1],min = 1e-6))) -
            torch.atan((boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1],min = 1e-6))), 2)
    alpha = v / (1-iou+v)


    #cal ciou
    cious = iou - (center_dis / outer_diagonal_line + alpha*v)

    return cious


def NMS(boxes,conf_thresh,iou_thresh):
  '''
  boxes format (batch_size , all boxes over all grids and anchors in all scales, 5+ number of classes)
  '''
  #boxes=boxes.to(device='cuda')
  mask=boxes[...,4]>conf_thresh

  batch_size=boxes.shape[0]
  output=[]
  for i in range(batch_size):
      batch   = boxes[i][mask[i]] ### (numboxes,5 + nclasses)
      ind     = torch.argsort(batch[...,4],dim=0) ###(numboxes,)
      batch   = batch[ind]

      batch[...,5:]= batch[...,5:] * batch[...,4:5] ## (numboxes,5 + nclasses)
      conf,j       = batch[...,5:].max(1,keepdim=True) ## (numboxes,1) (numboxes,1)
      batch        = torch.cat((batch[...,:4],conf,j),1)[conf[...,0] >conf_thresh] #(num_boxes,xywh + conf +classid)

      batch[...,:4] = xywh2xyxy(batch[..., :4])
      c  = batch[:, 5:6] * 500 # see https://github.com/ultralytics/yolov5/discussions/5825 for more details
      out_nms = torchvision.ops.nms(batch[...,:4]+c,batch[...,4],iou_thresh) # (numboxes,)
      output.append(batch[out_nms])
  return output

def scale_boxes(boxes, gain_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    #print(ratio_pad)
    gain = gain_pad[0]
    pad = gain_pad[1]
    #print(gain,ratio_pad)
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    
    return boxes

class get_color:
    def __init__(self,classes):
        self.color_map = dict()
        for i in classes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.color_map[i] = color
                
        
    def get_color(self, class_name):
        return self.color_map[class_name]
 
def draw_boxes(img_path, preds, shapes, classes):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
  colors = get_color(classes)
  preds=preds[0].cpu().numpy()
  for pred in preds:
    pred=abs(pred)
    if not pred.shape[0]:
      continue
    if len(classes) == 1:
      pred[5] = 0
    box = scale_boxes(pred[:4], img.shape[:2], shapes)
    cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),colors.get_color(classes[int(pred[5])]),2)
    cv2.putText(img,classes[int(pred[5])],(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_PLAIN, 3, colors.get_color(classes[int(pred[5])]),2)
  plt.imshow(img)
  

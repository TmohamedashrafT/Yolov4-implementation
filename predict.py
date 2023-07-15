from utils import NMS,load_image
import torch
import os
import numpy as np
import cv2

def pred_without_eval(img_path='',
                      model=None,
                      conf_thresh=0.1,
                      iou_thresh=0.6,
                      classes=[],
                      save_dire='/content',
                      device='cuda',
                      weights=None,
                      img_size=416):
  if weights:
    ckpt=torch.load(weights,map_location='cpu')
    model.load_state_dict(ckpt['weights'])
    del ckpt 
  img, (oldh, oldw), (newh,neww),scale,(dw,dh)=load_image(img_path,img_size)
  shapes  = scale, (dw,dh)
  img = img.transpose((2, 0, 1))[::-1]
  img = np.ascontiguousarray(img)
  img = torch.from_numpy(img[None])
  img = img.to(device, non_blocking=True).float()/255
  with torch.no_grad():
     preds=model(img)
    
  preds = NMS(preds,conf_thresh,iou_thresh)
  draw_boxes(img_path,preds,shapes,classes,save_dire)
def draw_boxes(img_path,preds,shapes,classes,save_dir=''):
  img = cv2.imread(img_path)    
  preds=preds[0].cpu().numpy()
  for pred in preds:
    #print(pred.shape)
    if not pred.shape[0]:
      continue
    if len(classes) == 1:
      pred[5] = 0
    box = scale_boxes( pred[:4], shapes)
    img = cv2.putText(img,str(classes[int(pred[5])])+' '+str(np.round(pred[4],2)),(int(box[0]), int(box[1]-10)),cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0), 2, cv2.LINE_AA)
    img = cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color=(0,0,255),thickness=2)
 
  di=img_path.strip().split('/')[-1]
  if save_dir[-1]!='/':
    save_dir=save_dir+'/' 
  cv2.imwrite(save_dir+di, img) 
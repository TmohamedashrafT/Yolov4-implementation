from utils import bb_iou
import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)

        return loss

class ComputeLoss:

    def __init__(self, device,
                 num_classes, 
                 anchors    =(),
                 balance    =[4.0,1.0,0.4],
                 iou_thresh =0.5,
                 img_size   =416
                 ):
        self.FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")
        self.balance        = balance
        self.BCE            = nn.BCEWithLogitsLoss(reduction='none')
        self.num_classes    = num_classes 
        self.device         = device
        self.anchors        = anchors.to(device=self.device)
        self.iou_thresh     = iou_thresh
        self.img_size       = img_size
    def __call__(self, predictions, targets, all_boxes): 
        cls_loss = torch.zeros(1, device=self.device)  
        box_loss = torch.zeros(1, device=self.device)  
        obj_loss = torch.zeros(1, device=self.device)  
        for i, pred in enumerate(predictions):  
            y_true  = targets[i].clone().to (device=self.device)
            boxes   = all_boxes[i].clone().to(device=self.device)

            batch_size = y_true.shape[0]
            pred_xy, pred_wh, pred_cls = pred[...,0:2],pred[...,2:4],pred[...,5:]  
            true_box,conf=y_true[...,0:4],y_true[...,4]
                
            ### boxes loss
            pred_xy   = pred_xy.sigmoid()
            pred_wh   = (torch.exp(pred_wh)) * self.anchors[i]
            pred_box  = torch.cat((pred_xy, pred_wh), -1)  
            ciou      = bb_iou(pred_box, true_box,fun='ciou')
            bbox_loss_scale = 2.0 - 1.0 * true_box[..., 2:3] * true_box[..., 3:4] / (self.img_size ** 2)
            box_loss += ((((1.0 - ciou) * conf * bbox_loss_scale[...,0]).sum()) / batch_size)

            #### obj loss
            iou = bb_iou(pred_box.unsqueeze(4), boxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
            iou_max = torch.max(iou,-1)[0]
            label_noobj_mask = (1.0 - conf) * (iou_max < self.iou_thresh).float()

            loss_conf= conf * (self.BCE(pred[...,4], target=conf) + label_noobj_mask * self.BCE(pred[...,4], target=conf))
            obj_loss += (loss_conf.sum() / batch_size) 

            #### classes loss    
            if self.num_classes > 1:  
                true_cls=y_true[...,5:]
                cls_loss += (self.BCE(pred_cls,true_cls).sum() / batch_size) 

            
        return (box_loss + obj_loss + cls_loss) , torch.cat((box_loss, obj_loss, cls_loss)).detach()
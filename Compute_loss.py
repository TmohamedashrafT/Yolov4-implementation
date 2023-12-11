import torch
import torch.nn as nn
from utils import bb_iou
class ComputeLoss:
    def __init__(self, device, num_classes, img_size):
        self.BCE            = nn.BCELoss(reduction="none")
        self.num_classes    = num_classes
        self.device         = device
        self.img_size       = img_size
    def __call__(self, preds, targets):
        tot_cls_loss = torch.zeros(1, device=self.device)
        tot_box_loss = torch.zeros(1, device=self.device)
        tot_obj_loss = torch.zeros(1, device=self.device)
        for i, pred in enumerate(preds):
            y_true  = targets[i].clone().to(device=self.device)
            pred_box, pred_conf, pred_cls = pred[...,:4], pred[...,4], pred[...,5:]
            true_box, true_conf, true_cls = y_true[...,:4],y_true[...,4], y_true[...,5:]

            ciou    = bb_iou(pred_box, true_box,fun='ciou')
            bbox_loss_scale = 2.0 - 1.0 * true_box[...,2:3] * true_box[...,3:4] / (self.img_size ** 2)
            tot_box_loss += ((1.0 - ciou) * true_conf * bbox_loss_scale[...,0]).sum(axis = [1,2,3]).mean()

            #### obj loss
            #conf_focal = (conf - pred[...,4]) ** 2
            obj_loss = self.BCE(pred_conf.float(), target = true_conf.float()) 
            tot_obj_loss +=  obj_loss.sum(axis = [1,2,3]).mean()

            #### classes loss
            if y_true.shape[-1] > 6:
                cls_loss = true_conf.unsqueeze(-1) * (self.BCE(pred_cls.float(),target = true_cls.float()))
                tot_cls_loss +=  cls_loss.sum(axis = [1,2,3]).mean()
        
        return (tot_box_loss + tot_obj_loss + tot_cls_loss) , torch.cat((tot_box_loss, tot_obj_loss, tot_cls_loss)).detach()
   

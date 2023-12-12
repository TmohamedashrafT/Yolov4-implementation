from utils import *
from Yolov4_architecture.model import Yolov4 
from Dataset_Generator import data_loader
import torch
import numpy as np
import argparse

def ap_per_class(tp, conf, pred_cls,count_classes):

    # Sort by objectness
    i = np.argsort(
        -conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]


    unique_classes=np.arange(0,len(count_classes))
    num_classes = len(count_classes)

    ap   = np.zeros((num_classes, tp.shape[1]))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = count_classes[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (count_classes[ci] + 1e-7)  # recall curve

        precision = tpc / (tpc + fpc)  # precision curve
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return  ap, unique_classes.astype(int)


def compute_ap(recall, precision):

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    return ap


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    ### n m
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = bb_iou(labels[:, 1:].unsqueeze(1), detections[:, :4].unsqueeze(0),format='xyxy')
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().detach().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def eval_(
        weights=None,
        batch_size=16,
        img_size=224,
        conf_thres=0.01,
        iou_thres=0.45,
        device='',
        model=None,
        val_loader=None,
        classes={},
        compute_loss=True,
        count_classes=[]

):

    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    nc=3
    seen = 0
    map50, ap50, map = 0.0, 0.0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (im, targets) in enumerate(val_loader):

        im = im.to(device, non_blocking=True).float()/255
        targets = targets.to(device)

        _, _, height, width = im.shape

        with torch.no_grad():
            preds = model(im)
        targets[...,1:] *= torch.tensor((img_size, img_size, img_size, img_size), device=device)
        preds = NMS(preds,conf_thres,iou_thres)


        for si, pred in enumerate(preds):
            labels = targets[si]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions

            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device)))
                continue
           # if True:
               #pred[:, 5] = 2
            predn = pred.clone()

            labels[:,1:5] = xywh2xyxy(labels[:, 1:5])
            correct = process_batch(predn, labels, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5]))


    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
         ap, ap_class= ap_per_class(*stats,count_classes)
         ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
         map50, map =  ap50.mean(), ap.mean()
    print('instances', seen,'   mAP@IoU 0.50 = ' ,np.round(map50,3),'  mAP@IoU 0.50:0.95 = ', np.round(map,3),'\n')

    maps = np.zeros(nc) + map
    #print(ap_class)
    for i, c in enumerate(ap_class):
        #print('class ' , classes[i], '    AP = ',ap)
        maps[c] = ap[i]
    return ( map50, map), maps
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, default = "data/BBCD.yaml", help = "data.yaml")  
    parser.add_argument("--task", type = str,  default = "test", help = "evaluation on train, val or test set ")  
    parser.add_argument("--weights", type = str, default = "", help = "weights path")  
    parser.add_argument("--img_size", type = int, default = 224, help = "image size")  
    parser.add_argument("--batch_size", type = int, default = 16, help = "batch_size")  

    opt = parser.parse_args()
    data =  read_yaml(opt.data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt  = torch.load(opt.weights, map_location='cpu')
    model = Yolov4(num_classes = len(ckpt['classes']),
                   anchors = ckpt['anchors'],
                   device  = device).to(device = device)
    model.load_state_dict(ckpt['weights'])
    classes = ckpt['classes'] 
    model.load_state_dict(ckpt['weights'])
    val_loader , count_classes = data_loader(img_dir    = data[f'{opt.task}_image_path'],
                                             ann_dir    = data[f'{opt.task}_ann_path'],
                                             anchors    = ckpt['anchors'],
                                             img_size   = opt.img_size,
                                             num_classes= len(ckpt['classes']),
                                             batch_size = opt.batch_size)
    map, maps  = eval_(device   = device,
                     batch_size = opt.batch_size,
                     img_size   = opt.img_size,
                     model      = model,
                     val_loader = val_loader,
                     classes    = classes,
                     count_classes = count_classes)  
    del ckpt
    

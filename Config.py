from easydict import EasyDict as edict
import torch
cfg   = edict()

 
    
### model information
cfg.model=edict()

cfg.model.device              ='cuda' if torch.cuda.is_available() else 'cpu'
cfg.model.train_image_path    =""
cfg.model.train_ann_path      =""
cfg.model.feature_channels    =[64, 128, 256, 512, 1024]
cfg.model.img_size            =416
cfg.model.stride              =[8,16,32]
cfg.model.batch_size          =16
cfg.model.data_aug_train      =True
cfg.model.data_aug_val        =False
cfg.model.optimizer           ='SGD'
cfg.model.num_classes         =1
cfg.model.anchors             =torch.tensor([[[ 1.2500,  1.6250],[ 2.0000,  3.7500],[ 4.1250,  2.8750]],
                                             [[ 1.8750,  3.8125],[ 3.8750,  2.8125],[ 3.6875,  7.4375]],
                                             [[ 3.6250,  2.8125],[ 4.8750,  6.1875],[11.6562, 10.1875]]])
cfg.model.head_scales         =[4.0,1.0,0.4]
cfg.model.lr                  =0.001
cfg.model.momentum            =0.937
cfg.model.weight_decay        =0.0005
cfg.model.box_loss_scale      =0.05
cfg.model.obj_loss_scale      =0.08 
cfg.model.cls_loss_scale      =0.1
cfg.model.iou_thresh          =0.5
cfg.model.iou_wh_anchor       =0.2
cfg.model.freeze              =[0]
cfg.model.start_epoch         =0
cfg.model.epochs              =50
cfg.model.patience            =10
cfg.model.weights_path        =""
cfg.model.classes             =['letter']
cfg.model.val_image_path      =""
cfg.model.val_ann_path        =""

#### for augmentation
cfg.aug = edict()
cfg.aug.RandomCrop = 0.5
cfg.aug.HorizontalFlip = 0.5
cfg.aug.RandomBrightnessContrast = 0.1
cfg.aug.VerticalFlip = 0.5
cfg.aug.MedianBlur = 0.2
cfg.aug.Blur = 0.1
cfg.aug.ImageCompression = 0.0
cfg.aug.Affine_scale = 0.1
cfg.aug.Affine_shear = 10
cfg.aug.Affine_rotate = (-10,10)
cfg.aug.Affine = 0.5




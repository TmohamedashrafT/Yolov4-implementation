import albumentations as A
import cv2
import numpy as np
import random

def augs(img,labels, class_labels, cfg_aug):
    transform = A.Compose([
    A.HorizontalFlip(p = cfg_aug['HorizontalFlip']),
    A.RandomBrightnessContrast(p = cfg_aug['RandomBrightnessContrast']),
    A.VerticalFlip(p = cfg_aug['VerticalFlip']),
    A.MedianBlur(p = cfg_aug['MedianBlur']),
    A.ImageCompression(quality_lower = 75, p = cfg_aug['ImageCompression']),
    A.Affine(scale=cfg_aug.Affine['scale'], shear = cfg_aug['Affine']['shear'], rotate = cfg_aug['Affine']['rotate'], p = cfg_aug['Affine']['p'])],
    bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels']))
    transformed = transform(image = img, bboxes = labels, class_labels = class_labels)
    img = transformed['image']
    labels=np.array([[c, *b] for c, b in zip(transformed['class_labels'], transformed['bboxes'])])
    return img, labels

def random_crop(self,image, bboxes):
        if random.random()<0.5:
            h,w,_=image.shape
            min_max_bbox=np.concatenate([np.min(bboxes[:,0:2],axis=0),np.max(bboxes[:,2:4],axis=0)],axis=0)
            
            min_box_x=min_max_bbox[0]
            min_box_y=min_max_bbox[1]
            max_box_x=w - min_max_bbox[2]
            max_box_y=h - min_max_bbox[3]
            
            crop_xmin=max(0,int(min_max_bbox[0] - random.uniform(0,min_box_x)))
            crop_ymin=max(0,int(min_max_bbox[1] - random.uniform(0,min_box_y)))
            crop_xmax=max(w,int(min_max_bbox[2] + random.uniform(0,max_box_x)))
            crop_ymax=max(h,int(min_max_bbox[3] + random.uniform(0,max_box_y)))
            
            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return image,bboxes               
def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


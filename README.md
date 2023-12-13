# Yolov4-implementation
Implementing the training pipeline for YOLOv4 using PyTorch

## Installation
1.Clone the repository
```
git clone https://github.com/TmohamedashrafT/Yolov4-implementation.git
```
2.Create a yaml file containing the data paths in the following format
```
train_image_path: 'train images path'
train_ann_path  : 'train annotations path'
val_image_path  : 'val images path'
val_ann_path    : 'val annotations path'
test_image_path : 'test images path' #can be empty
test_ann_path   : 'test annotations path' #can be empty
num_classes     : 'number of classes'
classes         :  ['name of class 1', 'name of class 2', ..., num_classes]
```
The annotation file must be in a txt file and in the following format
```
#yolo format
class_id, x_center, y_center, width, height
```
3.Train
```
%cd Yolov4-implementation 
!python Training.py --data data.yaml --hyp yamls/hyp.yaml --aug yamls/aug.yaml
```
The settings for augmentation and hyperparameters can be modified in hyp.yaml and aug.yaml.

4.evaluate
```
python eval.py --data data.yaml --task 'train' --weights 'best.pt'
```
5.predict
```
!python predict.py --src 'test_image_path'  --weights 'best.pt' --conf_thresh 0.4
```
This notebook is an example of training the model on the BCCD dataset [`Yolov4 test on BCCD-794 dataset.ipynb`](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/Yolov4%20test%20on%20BCCD-794%20dataset.ipynb)




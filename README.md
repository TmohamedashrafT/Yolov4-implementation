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

## Yolov4 Overview
The authors of the paper aimed to develop and model different approaches to enhance the YOLOv4 model, introducing numerous new features compared to previous versions. These enhancements encompassed modifications to the model's architecture, the loss function, and the incorporation of new augmentation methods that increased both speed and accuracy.

### Yolov4 Architecture
The YOLOv4 architecture consists of three main parts: the backbone, neck, and head.
#### Backbone
In the first part, the backbone serves as a feature extractor responsible for extracting essential features from the inputs. The paper suggests various backbones such as VGG16, ResNet-50, EfficientNet-B0/B7, and CSPDarknet53(which is implemented in this repository)
The idea behind CSPDarknet53 is rooted in Densenet but with modifications. The term 'CSP' stands for Cross Stage Partial, which signifies that in every dense block, the feature maps of the base layer (the first one) are separated into two parts. One part traverses through the block, while the other combines with the transition layer, allowing the rest of the dense block to continue as a normal dense layer. This block offers several benefits. Firstly, by separating the feature maps, it reduces the input size of the block, consequently reducing computations. Another significant advantage is the elimination of duplicated gradients in the old dense block because the output of the block concatenates with the transition layer. Additionally, it helps mitigate the issue of vanishing gradients.
![image](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/readme_imgs/CSPnet%20vs%20densenet2.png)
#### Neck
The Neck in YOLOv4 comprises two primary parts: additional blocks like SPP (Spatial Pyramid Pooling) and path-aggregation blocks such as PANet (Path Aggregation Network). In YOLOv4, SPP employs pooling sizes of 5x5, 9x9, and 13x13, maintaining the output spatial dimension. YOLOv4 adopts PAN (Path Aggregation Network) over the previously used FPN (Feature Pyramid Network) for feature aggregation. The issue with FPN stemmed from its top-down approach along the path from low to top layers. PAN addresses this by integrating a bottom-up block into the FPN architecture, creating a more direct path to the top layers. The diagram below illustrates this adjustment (green path).
![image](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/readme_imgs/FPN%20vs%20PANet.webp)










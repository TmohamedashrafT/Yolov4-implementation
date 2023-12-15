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
## TODO
* [x] Mish activation
* [x] CIOU Loss
* [x] Training code
* [x] Modified PAN
* [x] Class label smoothing
* [ ] Mosaic
* [ ] MixUp
## Yolov4 Overview
The authors of the paper aimed to develop and model different approaches to enhance the YOLOv4 model, introducing numerous new features compared to previous versions. These enhancements encompassed modifications to the model's architecture, the loss function, and the incorporation of new augmentation methods that increased both speed and accuracy.

### Yolov4 Architecture 
[`Yolov4_architecture`](https://github.com/TmohamedashrafT/Yolov4-implementation/tree/main/Yolov4_architecture)

The YOLOv4 architecture consists of three main parts: the backbone, neck, and head.
#### 1.Backbone
[`backbone`](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/Yolov4_architecture/backbone.py)

In the first part, the backbone serves as a feature extractor responsible for extracting essential features from the inputs. The paper suggests various backbones such as VGG16, ResNet-50, EfficientNet-B0/B7, and CSPDarknet53(which is implemented in this repository)
The idea behind CSPDarknet53 is rooted in Densenet but with modifications. The term 'CSP' stands for Cross Stage Partial, which signifies that in every dense block, the feature maps of the base layer (the first one) are separated into two parts. One part traverses through the block, while the other combines with the transition layer, allowing the rest of the dense block to continue as a normal dense layer. This block offers several benefits. Firstly, by separating the feature maps, it reduces the input size of the block, consequently reducing computations. Another significant advantage is the elimination of duplicated gradients in the old dense block because the output of the block concatenates with the transition layer. Additionally, it helps mitigate the issue of vanishing gradients.

![image](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/readme_imgs/CSPnet%20vs%20densenet2.png)
#### 2.Neck
[`SPP_PaNet`](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/Yolov4_architecture/SPP_PaNet.py)

The Neck in YOLOv4 comprises two primary parts: additional blocks like SPP (Spatial Pyramid Pooling) and path-aggregation blocks such as PANet (Path Aggregation Network). In YOLOv4, SPP employs pooling sizes of 5x5, 9x9, and 13x13, maintaining the output spatial dimension. YOLOv4 adopts PAN (Path Aggregation Network) over the previously used FPN (Feature Pyramid Network) for feature aggregation. The issue with FPN stemmed from its top-down approach along the path from low to top layers. PAN addresses this by integrating a bottom-up block into the FPN architecture, creating a more direct path to the top layers. The diagram below illustrates this adjustment (green path).

![image](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/readme_imgs/FPN%20vs%20PANet.webp)
#### 3.Head
[`head`](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/Yolov4_architecture/head.py)

The YOLOv4 head operates similarly to a standard YOLO3 head. For every grid in the network's output, it predicts a bounding box (x_center, y_center, width, height), determines the confidence score representing the model's confidence in this box, and predicts the probability of the object class within the box.

![image](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/readme_imgs/head.png)

#### Data loader
[`Dataset_Generator`](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/Dataset_Generator.py)

In the Dataset_Generator, the process begins by loading an image, then resizing it based on the image size. If data_augmentation is enabled (data_aug is True), augmentation techniques are applied to the image. Subsequently, the labels for this image are resized to fit the new image shape. During the training phase of the model, the labels are further processed to match the desired scale and anchor box, considering the aspect ratio of each object in the image individually. This involves calculating the Intersection over Union (IOU) between the anchor boxes at each scale and the label. A threshold is applied to the IOU values, assigning the label to the grids and anchor boxes based on the scale and position (x, y) if the IOU value exceeds the threshold. Notably, a label may belong to multiple scales, positions, and anchors. If the label doesn't match all anchors across scales, the class is assigned based on the maximum IOU it achieves. The labels follow the format (x_center, y_center, width, height) with range values from 0 to the image shape during training. During evaluation, the class passes labels without modifications in the same format, but with a range from 0 to 1.

#### Loss
[`Compute_loss`](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/Compute_loss.py)

In YOLOv4, the authors introduced a new Bounding Box regression loss, replacing the traditional Mean Squared Error (MSE) loss from previous versions with the Ciou loss. This loss is based on the Distance-IoU (DIoU), which calculates the distance between the centers of the predicted box and the ground truth, normalized by the diagonal length of the smallest enclosing box covering the two boxes. The Ciou loss incorporates an essential concept: the loss in aspect ratio. It prioritizes between the overlap and the aspect ratio, controlled by the parameter alpha in the equation. When the IoU is less than 0.5, alpha is zero, but when it's greater than 0.5, more attention is directed towards the aspect ratio

![image](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/readme_imgs/CIOU.png)

The "ComputeLoss" class computes the Ciou between the predicted boxes and ground truth by utilizing the "bb_iou" function. Afterward, it calculates the "bbox_loss_scale" to allocate more attention to small objects. It then proceeds to sum the loss over all grids and anchors, subsequently obtaining the mean of all batches.

In the original implementation, the objective loss is divided into two parts. Firstly, if an object exists in the ground truth, the loss computes binary cross-entropy between the ground truth (equal to 1) and the predicted confidence object, which ranges from 0 to 1.
Secondly, a mask is computed to identify boxes that don't surpass the IoU threshold (usually set to 0.3) with any other box in the image. These identified boxes are considered as containing no object and are penalized in the loss computation.

![image](https://github.com/TmohamedashrafT/Yolov4-implementation/blob/main/readme_imgs/objective%20loss.png)

In this repository, the objective loss is more stringent as it includes all boxes in the loss calculation without excluding any, leading to the need for more epochs to converge.

The class loss is straightforward, involving Binary Cross-Entropy (BCE) computation between the predicted and ground truth classes.

# References 
- https://arxiv.org/abs/2004.10934
- https://github.com/ultralytics/yolov5
- https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3


























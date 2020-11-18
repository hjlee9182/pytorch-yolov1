## pytorch YOLO-v1

This repository base on : https://github.com/xiongzihua/pytorch-YOLO-v1

**This is a experimental repository, which are not exactly the same as the original [paper](https://arxiv.org/pdf/1506.02640.pdf), our performance on coco2017 val is 28.65 map**

I write this code for the purpose of learning. In yoloLoss.py, i write forward only, with autograd mechanism, backward will be done automatically.

For the convenience of using pytorch pretrained model, our backbone network is resnet50, add an extra block to increase the receptive field, in addition, we drop Fully connected layer.

Effciency has not been optimized. It may be faster... I don't know 

![](person_result.jpg)

![](dog_result.jpg)

## Train on COCO train 2017
| model                | backbone | map@coco  |
| -------------------- | -------------- | ---------- |
| our ResNet_YOLO  |   coco        | 28.65%     |

### 1. Dependency
- pytorch 1.7 ( I think it's working also below 1.7 but I didn't test yet )
- opencv

### 2. Prepare

1. Download COCO 2017 image data and make txt file like "coco_train_vocstyle.txt"

txt file : path xmin ymin xmax ymax class_num

### 3. Train
Run python train.py

*Be careful:* 
1. change the image file path
2. check output file( .pth ) name

### 4. Evaluation
Run python predict.py

*be careful* 
1. change the image file path

If you want to calculate mAP you need to clone https://github.com/Cartucho/mAP

|

|---- pytorch-yolov1

|---- mAP



### 5. result

Our map in coco 2017 val set is 28.65%

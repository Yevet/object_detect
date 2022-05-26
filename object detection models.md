# object detection models



## hugging face(27 models):

> most models are mainly based on detr-resnet or yolo

### Based on facebook/detr-resnet

#### **1. facebook/detr-resnet-50 & facebook/detr-resnet-101 (AP: 42.0 & AP: 43.5 )**

pretrained on:  COCO 2017 object detection (118k annotated images and 92 object categories)

https://huggingface.co/facebook/detr-resnet-50 & https://huggingface.co/facebook/detr-resnet-101

https://arxiv.org/abs/2005.12872

https://github.com/facebookresearch/detr

#### **2. TahaDouaji/detr-doc-table-detection**

pretrained on ICDAR2019 Table Dataset

https://huggingface.co/TahaDouaji/detr-doc-table-detection

#### 3. **facebook/detr-resnet-50-dc5 (AP: 43.3)** 

pretrained on:  COCO 2017 object detection (118k annotated images and 92 object categories)

add dilated C5 stage 

https://huggingface.co/facebook/detr-resnet-50-dc5

#### **4. facebook/detr-resnet-101-dc5 (AP: **44.9 )

pretrained on:  COCO 2017 object detection (118k annotated images and 92 object categories)

add dilated C5 stage 

#### 5. davanstrien/detr_beyond_words (recognize different part in newspaper)

https://huggingface.co/davanstrien/detr_beyond_words

[facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) fine tuned on [Beyond Words](https://github.com/LibraryOfCongress/newspaper-navigator/tree/master/beyond_words_data).



## Based on facebook/detr-resnet

#### 1. **hustvl/yolos-small (AP: 36.1)**

pretrained on：[ImageNet-1k](https://huggingface.co/datasets/imagenet2012) (1000 object classes and contains 1,281,167 training images, 50,000 validation images )

https://huggingface.co/hustvl/yolos-small

https://arxiv.org/abs/2106.00666

https://github.com/hustvl/YOLOS

#### 2. hustvl/yolos-base(AP: 42.0)

pretrained on:  pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet2012) and fine-tuned on [COCO 2017 object detection](https://cocodataset.org/#download), a dataset consisting of 118k/5k annotated images for training/validation

https://huggingface.co/hustvl/yolos-base

#### 3. **hustvl/yolos-base**(AP: 28.7)

pretrained on:  pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet2012) and fine-tuned on [COCO 2017 object detection](https://cocodataset.org/#download), a dataset consisting of 118k/5k annotated images for training/validation

#### **4. hustvl/yolos-small-dwr**(AP: 42.0)

pretrained on:  pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet2012) and fine-tuned on [COCO 2017 object detection](https://cocodataset.org/#download), a dataset consisting of 118k/5k annotated images for training/validation

https://huggingface.co/hustvl/yolos-small-dwr



### Others

#### 1. mindee/fasterrcnn_mobilenet_v3_large_fpn(recognize word)

pretrained on [DocArtefacts](https://mindee.github.io/doctr/datasets.html#doctr.datasets.DocArtefacts).

https://huggingface.co/mindee/fasterrcnn_mobilenet_v3_large_fpn

#### 2. mishig/tiny-detr-mobilenetsv3

https://huggingface.co/mishig/tiny-detr-mobilenetsv3



------------------------------------------

#### github:

#### 1.TensorFlow Object Detection API 

https://github.com/tensorflow/models/tree/master/research/object_detection 

pretrained on Mobile Net COCO dataset

#### 2.faster_rcnn_resnet50_fpn 

pretrained on the Bosch Dataset

https://github.com/Opletts/Object-Detection-Labeller

#### 3.[ObjectDetectionTests](https://github.com/frankynavar/ObjectDetectionTests)

ssd_mobilenet_v1_coco_2017_11_17.tar.gz

faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz

ssd_inception_v2_coco_2017_11_17.tar.gz

rfcn_resnet101_coco_2018_01_28.tar.gz

#### 4.https://github.com/yehengchen/Object-Detection-and-Tracking 

 application of yolov3, R-CNN 






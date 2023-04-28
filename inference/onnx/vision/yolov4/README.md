## README Yolov4

# 目录

<!-- TOC -->

- [目录](#目录)
    - [Yolov4描述](#yolov4描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [模型文件](#模型文件)
    - [目录结构](#目录结构)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
        - [安装环境依赖](#安装环境依赖)
        - [设置common路径](#设置common路径)
        - [评估过程](#评估过程)
            - [评估](#评估)
    - [模型描述](#模型描述)
        - [模型精度](#模型精度)
    - [随机情况说明](#随机情况说明)

<!-- /TOC -->

## Yolov4描述

2020 年 4 月，AlexeyAB 在 YOLOv3 的基础上不断进行改进和开发，发布了 YOLOv4，并得到了原作者 Joseph Redmon 的认可。YOLOv4中结合了WRC, CSP, CmBN, SAT, Mish激活函数，Mosaic数据增强，CmBN，DropBlock，CIoU等新特征。与其他最先进的目标检测方法比较，YOLOv4 的运行速度是 EfficientDet 的两倍。AP 和 FPS 分别比 YOLOv3 提高了 10% 和 12%。

[论文](https://arxiv.org/pdf/2004.10934.pdf)： Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao. YOLOv4: Optimal Speed and Accuracy of Object Detection. *Computer Vision and Pattern Recognition*. 2020.

## 模型架构

YOLOv4的网络结构主要分为BackBone、Neck、Predection三部分组成，在BackBone部分YOLOv4使用了CSPDarkNet53结构，主要是由最小组件CBM（Conv+Bn+Mish）和CBL（Conv+Bn+Leaky_relu）堆叠而成，Neck部分YOLOv4采用SPP+PAN的结构，Predection部分有三个head，head是由CBL+Conv组成的。

## 数据集

使用的数据集：[COCO-2017](<https://cocodataset.org/#download>)

- 下载数据集：

```bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

## 模型文件

- 下载yolov4模型放在./model中。

[yolov4-leaky-608-darknet-op13-fp32-N.onnx](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/yolo/yolov4-leaky-608-darknet-op13-fp32-N.onnx?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1671178405;2535178405&q-key-time=1671178405;2535178405&q-header-list=&q-url-param-list=&q-signature=34fad7e2550c12a47e07d3b74045ed709b1390cb)

## 目录结构

```
yolov4/
├── data
│    ├── annotations
│    │   ├── captions_train2017.json
│    │   ├── captions_val2017.json
│    │   ├── instances_train2017.json
│    │   ├── instances_val2017.json
│    │   ├── person_keypoints_train2017.json
│    │   └── person_keypoints_val2017.json
│    └── val2017
│        ├── 000000000139.jpg
│        ├── 000000000285.jpg
│        ├── 000000000632.jpg
│        ├── ......
│        ├── 000000000776.jpg
│        └── 000000581781.jpg
├── data_processing.py
├── mdoel
│     └── yolov4-leaky-608-darknet-op13-fp32-N.onnx
├── README.md
├── requirements.txt
└── run_onnx.py
```

## 环境要求

- 硬件（GCU）
    - 准备GCU处理器搭建硬件环境。
- 框架
    - [Onnxruntime-TopsInference]

## 快速入门

通过官方网站安装Onnxruntime-TopsInference后，您可以按照如下步骤进行评估：

### 安装环境依赖

```shell
pip3 install -r requirements.txt
```

### 设置common路径

- 您需要将common文件夹的上级目录加入到PYTHONPATH环境变量

```shell
export PYTHONPATH=<parent/path/of/common>
```

也就是让PYTHONPATH路径呈现如下结构

```shell
PYTHONPATH
    ├── common
    └── vision/yolov4
                └── run_onnx.py
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- GCU处理器环境运行

  ```bash
  python3 run_onnx.py --dataset ./data --model ./model/yolov4-leaky-608-darknet-op13-fp32-N.onnx --device gcu --conf-thres 0.001 --iou-thres 0.6 --scale_x_y 1.05
  ```

- GPU环境运行

  ```bash
 python3 run_onnx.py --dataset ./data --model ./model/yolov4-leaky-608-darknet-op13-fp32-N.onnx --device gpu --conf-thres 0.001 --iou-thres 0.6 --scale_x_y 1.05
  ```

- CPU环境运行

  ```bash
 python3 run_onnx.py --dataset ./data --model ./model/yolov4-leaky-608-darknet-op13-fp32-N.onnx --device cpu --conf-thres 0.001 --iou-thres 0.6 --scale_x_y 1.05
  ```

- 可通过屏幕打印查看结果。测试数据集的测试结果打印格式如下：

```bash
 Final report:
{
    "model": "<path/to/onnx>",
    "dataset": "<path/to/dataset>"
    "batch_size": 1,
    "device": "gcu",
    "mAP": 0.xxxxx
}
```

## 模型描述

### 模型精度

| 参数 | GCU <br>（FP16混合精度）| GPU <br>（FP32）| CPU <br>（FP32）|
| :--------------------------: | :--------------------------: | :--------------------------: | :--------------------------: |
| 资源 | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | Nvidia A10 Tensor Core GPU, 24GB GDDR6显存; 系统 Linux | X86_64 Intel CPU, 2.10GHZ, 内存 32G; 系统 Linux|
| onnxruntime-topsinference版本 | 1.9.1 | 1.9.1 | 1.9.1 |
| 数据集 | COCO-val2017 | COCO-val2017 | COCO-val2017 |
| mAP | 0.490 | 0.490 | 0.490 |

## 随机情况说明

暂无

## README RESNET

# 目录

<!-- TOC -->

- [目录](#目录)
    - [Resnet描述](#resnet描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
        - [设置PYTHONPATH](#设置PYTHONPATH)
        - [运行](#运行)
    - [脚本说明](#脚本说明)
        - [脚本及样例代码](#脚本及样例代码)
        - [脚本参数](#脚本参数)
        - [评估过程](#评估过程)
            - [评估](#评估)
    - [模型描述](#模型描述)
        - [模型精度](#模型精度)
    - [随机情况说明](#随机情况说明)

<!-- /TOC -->

## Resnet描述

ResNet是一个图像分类模型，提出残差学习的概念来解决网络层数过深时较难训练的问题。
网络的深度对模型的性能至关重要，当增加网络层数后，网络可以进行更加复杂的特征模式的提取，理论上可以取得更好的结果，但是越深的网络越难训练，网络深度增加时，存在着梯度消失或者爆炸的可能，网络准确度可能出现饱和，甚至出现下降的现象。
Resnet提出残差学习的概念，把网络设计为H(x) = F(x) + x，即直接把恒等映射作为网络的一部分，把问题转化为学习一个残差函数F(x) = H(x) - x，减小学习的难度。

[论文](https://arxiv.org/abs/1512.03385)： Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition.

## 模型架构

ResNet由卷积层、BatchNorm层、Relu激活函数、池化层、若干残差结构、分类器构成。残差结构中，主干部分由两组Conv-BN-Relu和一组Conv-BN构成，支线为残差结构的输入，与主干的输出相加后再经过激活函数Relu。下采样时，主干部分第二个卷积步长为2，支线增加步长为2的卷积层。分类器包括全局平均池化层和全连接层。

## 数据集

使用的数据集：[Imagenet](https://image-net.org/)

- 数据集大小：14,197,122张图片
- 数据格式：jpeg
- 下载数据集，并按照[数据集准备说明](../../common/prepare_dataset/imagenet/README.md)处理成如下格式。 

```text
data
   ├── n01440764
   │   ├── ILSVRC2012_val_00000293.JPEG
   │   ├── ILSVRC2012_val_00002138.JPEG
   ……
   └── val_map.txt
```

## 模型文件

- ONNX模型：[下载链接](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/resnet/resnet50_v1.5-torchvision-op13-fp32-N.onnx?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1671178012;2535178012&q-key-time=1671178012;2535178012&q-header-list=&q-url-param-list=&q-signature=fea8580afc6e07643fdb36fa5f7602c22ece8178)
- 下载后放入model文件夹

## 环境要求

- 硬件（GCU）
    - 准备GCU处理器搭建硬件环境。
- 框架
    - [Onnxruntime-TopsInference]
- 依赖
    
    ```shell
    pip3 install pillow
    ```

## 快速入门

通过官方网站安装Onnxruntime-TopsInference后，您可以按照如下步骤进行评估：

### 设置PYTHONPATH

- 您需要将common文件夹的上级目录加入到PYTHONPATH环境变量

```shell
export PYTHONPATH=<parent/path/of/common>
```

也就是让PYTHONPATH路径呈现如下结构

```shell
PYTHONPATH
    ├── common
    └── vision/resnet
                └── run_onnx.py
```

### 运行

```bash
python3 run_onnx.py --model model/resnet50_v1.5-torchvision-op13-fp32-N.onnx --dataset=data/ --device=gcu
```

## 脚本说明

### 脚本及样例代码

```shell
./
├──run_onnx.py
└──README.md
```

### 脚本参数

```text
run_onnx.py中主要参数如下：

--model：onnx模型加载路径。
--dataset：数据集所在路径。
--device：实现代码的设备。可选值为"gcu"、"cpu"或"gpu"。
--batch_size：batch size。
--num_workers：加载数据集的worker数量。
--input_height：输入图片高度，默认224。
--input_width：输入图片宽度，默认224。
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- GCU环境运行

```bash
# FP16混合精度

python3 run_onnx.py --model model/resnet50_v1.5-torchvision-op13-fp32-N.onnx --dataset=data/ --device=gcu
```

```bash
# FP32精度
export ORT_TOPSINFERENCE_FP16_ENABLE=0
 
python3 run_onnx.py --model model/resnet50_v1.5-torchvision-op13-fp32-N.onnx --dataset=data/ --device=gcu
```

- CPU环境运行

```bash
python3 run_onnx.py --model model/resnet50_v1.5-torchvision-op13-fp32-N.onnx --dataset=data/ --device=cpu
```

可通过屏幕打印查看结果。测试数据集的测试结果打印格式如下：

```bash
 Final report:
 {
    "model": "<path/to/onnx>",
    "dataset": "<path/to/dataset>"
    "device": "gcu",
    "acc1": 0.xxxxx,
    "acc5": 0.xxxxx
 }
```

## 模型描述

### 模型精度

| 参数 | GCU <br>（FP16混合精度）| GCU <br>（FP32） | CPU <br>（FP32）|
| :--------------------------: | :--------------------------: | :--------------------------: | :--------------------------: |
| 资源 | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | X86_64 Intel CPU, 2.10GHZ, 内存 32G; 系统 Linux|
| 上传日期 | 2022-12-02 | 2022-12-02 | 2022-12-02 |
| onnxruntime-topsinference版本 | 1.9.1 | 1.9.1 | 1.9.1 |
| 数据集 | Imagenet | Imagenet | Imagenet |
| 测试精度 | FP16混合精度 | FP32 | FP32 |
| acc1 | 0.76126 | 0.7613 | 0.7613 |
| acc5 | 0.92862 | 0.92862 | 0.92862 |

## 随机情况说明

暂无
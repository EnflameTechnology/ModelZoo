## README CRNN

# 目录

<!-- TOC -->

- [目录](#目录)
    - [CRNN描述](#CRNN描述)
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

## CRNN描述

CRNN是一种文本识别模型，可以实现端到端地不定长的文本序列识别任务。
CRNN主要包含CNN、RNN和CTC模块。其中CNN特征提取层和RNN序列特征提取层，能够实现端到端的联合训练。CRNN的RNN和CTC模块可以学习文本上下的语义关系，提升文本识别准确率。
CNN特征提取层以常用的ResNet、MobileNet系列模型为主，能够提取比较优质的图像文本特征。
RNN序列特征提取层采用双向递归神经网络模型（LSTM），具有很强的序列上下文信息捕获能力。
CTC模块主要用于训练时的loss计算，将LSTM预测结果转成标签序列。

[论文](https://arxiv.org/pdf/1507.05717.pdf)： Baoguang Shi, Xiang Bai, Cong Yao. An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition.

## 模型架构

CRNN模型包含3个模块：CNN、LSTM和CTC。CNN采用ResNet34模型，由Conv、BN、Pooling及残差结构组成。LSTM采用双向RNN（BLSTM），一个向前的LSTM和一个向后的LSTM组合在一起，增强上下文信息捕获能力。CTC Loss用于将预测结果转录为最终的预测标签，主要用于模型训练。

## 数据集

数据集：[IIIT5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz)
标签：[test_label.txt](https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt)

- 数据集大小：3000张测试图片
- 数据格式：png

数据集放到文件夹 `./data` 下面。

数据集准备步骤如下：
1. 下载数据集：

``` shell
wget http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz
``` 
2. 解压数据集：

```shell
tar -xvf IIIT5K-Word_V3.0.tar.gz
```

3. 下载标签：

```shell
wget https://download.openmmlab.com/mmocr/data/mixture/IIIT5K/test_label.txt
```

4. 数据集结构

```shell
data
  ├── test
  │     ├── 1_1.png
  │     ├── 3_1.png
  ...
  └── test_label.txt
```

## 模型文件

- ONNX模型：[下载链接](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/crnn/crnn-resnet34-en-ppocr-op13-fp32-N.onnx?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1681371893;2041371893&q-key-time=1681371893;2041371893&q-header-list=&q-url-param-list=&q-signature=0f1fc50c6efa973aee526fa569035614161780a3)
- 下载后放入model文件夹

## 环境要求

- 硬件（GCU）
    - 准备GCU处理器搭建硬件环境。
- 框架
    - [Onnxruntime-TopsInference]
- 依赖
    
    ```shell
    pip3 install -r requirements.txt
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
    └── vision/crnn
                └── run_onnx.py
```

### 运行

```bash
python3 run_onnx.py --model ./model/crnn-resnet34-en-ppocr-op13-fp32-N.onnx --dataset ./data --device gcu
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
--device：实现代码的设备。可选值为"gcu"、"cpu"或"gpu"。
--dataset：数据集所在路径。
--batch_size：batch size。
--num_workers：加载数据集的worker数量。
--image_shape：模型输入图像尺寸。
--rec_char_dict_path：文本字典。
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- GCU环境运行

```bash
# FP16混合精度

python3 run_onnx.py --model ./model/crnn-resnet34-en-ppocr-op13-fp32-N.onnx --dataset ./data/ --device gcu
```

```bash
# FP32精度
export ORT_TOPSINFERENCE_FP16_ENABLE=0
 
python3 run_onnx.py --model ./model/crnn-resnet34-en-ppocr-op13-fp32-N.onnx --dataset ./data/ --device gcu
```

- CPU环境运行

```bash
python3 run_onnx.py --model ./model/crnn-resnet34-en-ppocr-op13-fp32-N.onnx --dataset ./data/ --device cpu
```

可通过屏幕打印查看结果。测试数据集的测试结果打印格式如下：

```bash
 Final report:
 {
    "model": "<path/to/onnx>",
    "dataset": "<path/to/dataset>"
    "device": "gcu",
    "acc": 0.xxxxx,
 }
```

## 模型描述

### 模型精度

| 参数 | GCU <br>（FP16混合精度）| GCU <br>（FP32） | CPU <br>（FP32）|
| :--------------------------: | :--------------------------: | :--------------------------: | :--------------------------: |
| 资源 | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | X86_64 Intel CPU, 2.10GHZ, 内存 32G; 系统 Linux|
| 上传日期 | 2023-03-17 | 2023-03-17 | 2023-03-17 |
| onnxruntime-topsinference版本 | 1.9.1 | 1.9.1 | 1.9.1 |
| 数据集 | Imagenet | Imagenet | Imagenet |
| 测试精度 | FP16混合精度 | FP32 | FP32 |
| acc | 0.8770 | 0.8770 | 0.8770 |

## 随机情况说明

暂无
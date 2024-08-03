## README DBNet

# 目录
<!-- TOC -->

- [目录](#目录)
    - [DBNet描述](#DBNet描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [模型文件](#模型文件)
    - [目录结构](#目录结构)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
        - [安装环境依赖](#安装环境依赖)
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

## DBNet描述

DBNet 基于分割进行文本检测，是一个较为常用的模型。
在一般基于分割的文本检测网络中，最终的二值化 map 都是使用的固定阈值来获取，并且阈值不同对性能影响较大。
本方法对每一个像素点进行自适应二值化，二值化阈值由网络学习得到，彻底将二值化这一步骤加入到网络里一起训练，这样最终的输出图对于阈值就会非常鲁棒。

[论文](https://arxiv.org/abs/1911.08947): Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, XiangLiao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang. Real-time Scene Text Detection with Differentiable Binarization. *Proc. AAAI*. 2020.

## 模型架构

DBNet 网络结构由 3 部分组成，分别为 backbone, neck, DB head.

- Backbone 有多种选择：resnet 系列有 resnet18, resnet34, resnet50, resnet101 等。还有 mobilenet, shufflenet 等。
- Neck 部分采用 FPN 结构。
- Head 部分为 DBNet, 由 Neck 的输出 feature map 预测得到 probability map 和 threshold map, 并根据二值化公式计算得到 近似二值图。

## 数据集

数据集放到文件夹 `./data` 下面。

数据集准备步骤如下：

1. 下载[ICDAR 2015 测试数据集](https://rrc.cvc.uab.es/?ch=4&com=downloads)(下载需要注册)。注册完登录后，下载 `Task 4.1: Text Localization (2015 edition)` 中的 `Test Set Images` 和 `Test Set Ground Truth`. 其中，`Test Set Images` 下载的内容保存到`ch4_test_images` 文件夹内，`Test Set Ground Truth` 放在 `ch4_test_localization_transcription_gt` 文件夹内。

2. 解压下载的压缩文件：

  ``` shell
  cd path/to/ch4_test_images
  unzip ch4_test_images.zip
  cd path/to/ch4_test_localization_transcription_gt
  unzip ch4_test_localization_transcription_gt.zip
  ```

3. 下载 [label 文件](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt)，将其放到 `ch4_test_images` 同目录下。

4. 下载 `https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json`，将其放到 `ch4_test_images` 同目录下，并执行`python3 modify_directory.py`.

## 模型文件

- ONNX模型：[下载链接](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/dbnet/dbnet-mv3-640x640-op13-fp32-N%281%29.onnx?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1681371317;2041371317&q-key-time=1681371317;2041371317&q-header-list=&q-url-param-list=&q-signature=926f77bea0f62cdfcfa9c57b2f25e389ca7d0f88)

- 下载后放入 `./model` 文件夹

## 目录结构

``` shell
model/
   └── dbnet-mv3-640x640-op13-fp32-N.onnx
data/
├── ch4_test_images
│        ├── img_1.jpg
│        ├── img_2.jpg
|        └── ...
├── ch4_test_localization_transcription_gt
│        ├── gt_img_1.txt
│        ├── gt_img_2.txt
|        └── ...
├── instances_test.json
└── test_icdar2015_label.txt
```

## 环境要求

- 硬件（GCU）
  - 准备 GCU 处理器，搭建硬件环境。

- 框架
  - onnxruntime-topsInference

## 快速入门

通过官方网站安装 onnxruntime-topsInference 后，您可以按照如下步骤进行评估：

### 安装环境依赖

``` shell
pip install -r requirements.txt
```

### 设置PYTHONPATH

- 您需要将 `common` 文件夹的上级目录加入到 `PYTHONPATH` 环境变量

``` shell
export PYTHONPATH=<parent/path/of/common>
```

也就是让 `PYTHONPATH` 路径呈现如下结构

``` shell
PYTHONPATH
    ├── common
    └── vision/ocr/dbnet
            ├── data_preprocess.py
            ├── eval_metric.py
            ├── post_process.py
            └── run_onnx.py
```

### 运行

- onnxruntime gcu 精度测试

``` shell
python3 run_onnx.py --model model/dbnet-mv3-640x640-op13-fp32-N.onnx --dataset=./data --device=gcu
```

- onnxruntime cpu 精度测试

``` shell
python3 run_onnx.py --model model/dbnet-mv3-640x640-op13-fp32-N.onnx --dataset=./data --device=cpu
```

## 脚本说明

### 脚本及样例代码

``` shell
dbnet/
├── data_preprocess.py
├── eval_metric.py
├── post_process.py
├── README.md
├── requirements.txt
└── run_onnx.py
```

### 脚本参数

``` bash
run_onnx.py 中主要参数如下：

--model：onnx 模型加载路径。
--dataset：数据集所在路径。
--device：运行代码的设备。可选值为 "gcu" "cpu" 或 "gpu"
--batch_size：batch size。
--input_height：输入图片高度，默认为 640。
--input_width：输入图片宽度，默认为 640。
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的 checkpoint 文件路径。

- GCU 环境运行

``` bash
# FP16 混合精度
python3 run_onnx.py --model model/dbnet-mv3-640x640-op13-fp32-N.onnx --dataset=./data --device=gcu
```

``` bash
# FP32 精度
export ORT_TOPSINFERENCE_FP16_ENABLE=0
python3 run_onnx.py --model model/dbnet-mv3-640x640-op13-fp32-N.onnx --dataset=./data --device=gcu
```

- CPU 环境运行

``` bash
python3 run_onnx.py --model model/dbnet-mv3-640x640-op13-fp32-N.onnx --dataset=./data --device=cpu
```

可通过屏幕打印查看结果。测试数据集的测试结果打印格式如下：

``` bash
 Final report:
 {
    "model": "<path/to/onnx>",
    "dataset": "<path/to/dataset>",
    "batch_size": 1,
    "device": "gcu",
    "precision": 0.xxxxx,
    "recall": 0.xxxxx,
    "hmean": 0.xxxxx
 }
```

## 模型描述

### 模型精度

| 参数 | GCU <br>（FP16混合精度）| GCU <br>（FP32） | CPU <br>（FP32）|
| :--------------------------: | :--------------------------: | :--------------------------: | :--------------------------: |
| 资源 | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | X86_64 Intel CPU, 2.10GHZ, 内存 32G; 系统 Linux|
| 上传日期 | 2023-03-08 | 2023-03-08 | 2023-03-08 |
| onnxruntime-topsinference版本 | 1.9.1 | 1.9.1 | 1.9.1 |
| 数据集 | ICDAR 2015 | ICDAR 2015 | ICDAR 2015 |
| 测试精度 | FP16混合精度 | FP32 | FP32 |
| precision | 0.3380 | 0.3409 | 0.3415 |
| recall | 0.0929 | 0.0939 | 0.0938 |
| hmean | 0.1458 | 0.1472 | 0.1472 |

## 随机情况说明

暂无

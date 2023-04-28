## README SWIN TRANSFORMER

# 目录

<!-- TOC -->

- [目录](#目录)
    - [SwinTransformer描述](#SwinTransformer描述)
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

## SwinTransformer描述

SwinTransformer是一个图像分类模型，是ViT的一个改进版本，主要提出了层次化特征映射和窗口注意力转换。
Transformer本身是一个长度序列不变的变换，使用在视觉领域时，会存在视觉特征尺度大、计算量大的问题。SwinTransformer提出Swin Transformer Block，实现了层级结构的Transformer模型，不同Block有不同的下采样倍数。此外，提出Window MSA (W-MSA)和shift Window MSA (SW-MSA)模块取代了ViT中使用的标准多头自注意力(MSA)模块，降低模型计算量，完成窗口间的信息交互。

[论文](https://arxiv.org/pdf/2103.14030.pdf)： Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin,  Baining Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.

## 模型架构

SwinTransformer包含4个stage，每个stage中包含2个部分：Patch Merging和Swin Transformer Block。Patch Merging类似于池化操作，但是不会有信息损失，用来完成特征下采样。Swin Transformer Block包含两个模块：W-MSA和SW-MSA。W-MSA在窗口内进行transformer操作，相对MSA方法，计算复杂度大幅度下降。SW-MSA用于弥补W-MSA的窗口间无法信息交互的问题。

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

- ONNX模型：[下载链接](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/swin/swin_base_patch4_window7_224-ms-op13-fp32-N.onnx?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1681372473;2041372473&q-key-time=1681372473;2041372473&q-header-list=&q-url-param-list=&q-signature=a2a82d55e82fe4a2eaa8a95e0c2afcc3169b131e)
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
    └── vision/swin_transformer
                    └── run_onnx.py
```

### 运行

```bash
python3 run_onnx.py --model ./model/swin_base_patch4_window7_224-ms-op13-fp32-N.onnx --dataset ./data --device gcu
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

--img_size：输入图像尺寸，默认224.
--model：onnx模型加载路径。
--dataset：数据集所在路径。
--device：实现代码的设备。可选值为"gcu"、"cpu"或"gpu"。
--batch_size：batch size。
--num_workers：加载数据集的worker数量。
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- GCU环境运行

```bash
# FP16混合精度

python3 run_onnx.py --model ./model/swin_base_patch4_window7_224-ms-op13-fp32-N.onnx --dataset ./data --device gcu
```

```bash
# FP32精度
export ORT_TOPSINFERENCE_FP16_ENABLE=0
 
python3 run_onnx.py --model ./model/swin_base_patch4_window7_224-ms-op13-fp32-N.onnx --dataset ./data --device gcu
```

- CPU环境运行

```bash
python3 run_onnx.py --model ./model/swin_base_patch4_window7_224-ms-op13-fp32-N.onnx --dataset ./data --device cpu
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
| 上传日期 | 2023-03-17 | 2023-03-17 | 2023-03-17 |
| onnxruntime-topsinference版本 | 1.9.1 | 1.9.1 | 1.9.1 |
| 数据集 | Imagenet | Imagenet | Imagenet |
| 测试精度 | FP16混合精度 | FP32 | FP32 |
| acc1 | 0.8342 | 0.8335 | 0.8342 |
| acc5 | 0.9644 | 0.9645 | 0.9645 |

## 随机情况说明

暂无
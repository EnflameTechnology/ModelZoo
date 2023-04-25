# README CONFORMER

[TOC]



# 目录

## CONFORMER描述

Conformer是一个谷歌于2020年提出的一个语音识别模型，该模型将Transformer结构和CNN结构在Encoder端融合到了一起，以此有效得抽取了音频的局部特征和全局特征，该模型在当时LibriSpeech测试集上取得了最好的效果

[论文]([[2005.08100\] Conformer: Convolution-augmented Transformer for Speech Recognition (arxiv.org)](https://arxiv.org/abs/2005.08100#))：Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang. Conformer: Convolution-augmented Transformer for Speech Recognition

## 模型架构

Conformer主要包含了以下3个模块，分别为Feedforward Module、Multi-head self attention Module以及Convolution Module.其中，Feedforward Module 包括了Layrnorm模块以及两个线性层的模块。Multi-head self attention Module包括了一个Layer Norm模块和一个采用相对位置的多头注意力模块。Convolution Module模块会先增加一个由逐点卷积和线性门控单元（gated linear unit，GLU）组成的门控机制，其后接一个一维的深度分离卷积，然后增加一个Batchnorm来帮助训练更深的模型。输入的音频首先转化为Fbank特征，然后依次进入上述的模块，最后再进入一次Feedforward Module，形成一个类似三明治的模型结构。

## 数据集

使用的数据集，下载及数据准备方法参考[数据准备章节](../../common/prepare_dataset/an4)：

- 数据集大小：23M
- 数据格式：wav
- 下载数据集，并按照数据集是准备说明处理成下格式

```bash
./an4 
|-- LICENSE
|-- README
|-- etc
|   |-- an4.dic
|   |-- an4.filler
|   |-- an4.phone
|   |-- an4.ug.lm
|   |-- an4.ug.lm.DMP
|   |-- an4_test.fileids
|   |-- an4_test.transcription
|   |-- an4_train.fileids
|   `-- an4_train.transcription
|-- test_manifest.json
|-- train_manifest.json
`-- wav
    |-- an4_clstk
    `-- an4test_clstk
```

## 模型文件

- ONNX模型：[下载链接](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/conformer/conformer_small-asr-nvidia-op13-fp32-N.onnx?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1681371582;2041371582&q-key-time=1681371582;2041371582&q-header-list=&q-url-param-list=&q-signature=c2f95cd71032d2c85b781d8da4c8328630f7766f) 
- nemo模型：[下载链接](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/conformer/stt_en_conformer_ctc_small.nemo?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1681372030;2041372030&q-key-time=1681372030;2041372030&q-header-list=&q-url-param-list=&q-signature=8591f10206c913964b4330d3e9a91b9b5ad0a015)
- 下载后放入model文件夹

## 环境要求

- 硬件（GCU）
  - 准备GCU处理器搭建硬件环境
- 框架
  - [Onnxruntime-TopsInference]
- 依赖

```bash
pip3 install -r requirements.txt
pip3 install nemo_toolkit['all']==1.5.0  --force-reinst
sudo apt-get install libsndfile1
pip3 install soundfile
```

## 快速入门

通过官方网站安装Onnxruntime-TopsInference后，您可以按照如下步骤进行评估：

### 设置PYTHONPATH

- 您需要将common文件夹的上级目录加入到PYTHONPATH环境变量

```bash
export PYTHONPATH=<parent/path/of/common>
```

也就是让PYTHONPATH路径呈现如下结构

```bash
PYTHONPATH
    ├── common
    └── speech/conformer
            └── run_onnx.py
```

### 运行

```bash
python3 run_onnx.py --model=model/conformer_small-asr-nvidia-op13-fp32-N.onnx --device=gcu --padding_mode=True --dataset
```

## 脚本说明

### 脚本及样例代码

```bash
conformer/
├── an4
├── model
│   ├── conformer_small-asr-nvidia-op13-fp32-N.onnx
│   └── stt_en_conformer_ctc_small.nemo
├── README.md
├── requirements.txt
└──  run_onnx.py
```

### 脚本参数

```bash
run_onnx.py中主要参数如下：

--model:onnx模型及预训练模型的加载路径
--dataset:an4数据集的json文件
--device:实现代码的设备，可选值为"gcu"、"cpu"或"gpu"
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径

- GCU环境运行

```bash
python3 run_onnx.py --model=model/conformer_small-asr-nvidia-op13-fp32-N.onnx --device=gcu --padding_mode=True --dataset <path of test_manifest.json>
```

- CPU环境运行

```bash
python3 run_onnx.py --model=model/conformer_small-asr-nvidia-op13-fp32-N.onnx --device=cpu --padding_mode=True --dataset <path of test_manifest.json>
```

可通过屏幕打印查看结果。测试数据集的测试结果打印格式如下：

```bash
{
    "python": "3.6.9",
    "tops-inference": "2.2.20230317",
    "onnxruntime": "1.9.1",
    "model": "./model/conformer_small-asr-nvidia-op13-fp32-N.onnx",
    "dataset": "./an4/test_manifest.json",
    "device": "gcu",
    "wer": 5.869230769230769
}

```

## 模型描述

### 性能

### 评估性能

| 参数                          | GCU                                           | CPU                                             |
| ----------------------------- | --------------------------------------------- | ----------------------------------------------- |
| 资源                          | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | X86_64 Intel CPU, 2.10GHZ, 内存 32G; 系统 Linux |
| 上传日期                      | 2023-03-20                                    | 2023-03-20                                      |
| onnxruntime-topsinference版本 | 1.9.1                                         | 1.9.1                                           |
| 数据集                        | an4                                           | an4                                             |
| 测试精度                      | FP32                                          | FP32                                            |
| cer                           | 5.87                                          | 5.87                                            |



## 随机情况说明

暂无
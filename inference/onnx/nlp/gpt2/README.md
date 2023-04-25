## README GPT-2

# 目录

<!-- TOC -->

- [目录](#目录)
    - [GPT-2描述](#GPT-2描述)
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
        - [性能](#性能)
            - [评估性能](#评估性能)
    - [随机情况说明](#随机情况说明)

<!-- /TOC -->

## GPT-2描述

GPT-2是一个Transformer 模型，以自监督的方式在大量英语数据集上进行了预训练。GPT-2是2018年的GPT的直接扩展，其在参数量和训练数据集上都扩展了10倍。GPT-2模型由多层单向 Transformer 的解码器部分构成。

[论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
Language Models are Unsupervised Multitask Learners


## 模型架构

GPT-2由embedding层、多个transformer layer、layernrom层组成。一个transformer层包含一个attention层、一个MLP层、两个layernorm和两个残差结构。在attention结构中，输入首先经过layernorm层，计算query、key、value矩阵，query和key进行矩阵乘得到attention score。attention score上加mask使得模型只关注左侧序列对输出的影响。value和attention score进行矩阵乘的到context tensor。context tensor后接一个矩阵乘和dropout。attention结构后接残差结构。在MLP结构中，输入首先经过layernorm层，然后经过一个矩阵乘，将tensor的维度扩展为4倍hiddensize。后接GELU，再经过一个矩阵乘，将tensor的维度恢复为hiddensize大小，后接dropout。MLP结构后依然是一个残差结构。

## 数据集

使用的数据集wikitext-2-raw-v1 test：[wikitext](https://huggingface.co/datasets/wikitext)

- 数据集大小：18M
- 数据格式：arrow
- 不需要预先下载数据集，执行评估脚本，会自动下载数据集。 目录结构如下：

```shell
data/
...
└── wikitext
```


## 模型文件

- ONNX模型：[下载链接](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/GPT/gpt2_small-huggingface-op13-fp32-seqN-nocache.onnx?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1681371449;2041371449&q-key-time=1681371449;2041371449&q-header-list=&q-url-param-list=&q-signature=143a3bbaf9de6b76c0b2f11b4d40a71dc5eaad75)
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
    └── nlp/gpt2
            └── run_onnx.py
```

### 运行

```bash
python3 run_onnx.py --model model/gpt2_small-huggingface-op13-fp32-seqN-nocache.onnx --batchsize 1 --dataset data --device gcu
```

## 脚本说明

### 脚本及样例代码

```shell
./
gpt2/
├── data
│   ...
│   └── wikitext
├── model
│   └── gpt2_small-huggingface-op13-fp32-seqN-nocache.onnx
├── README.md
├── requirements.txt
├── run_onnx.py
└── utils
    ├── create_batch_data.py
    └── onnx_text_generate.py
```

### 脚本参数

```text
run_onnx.py中主要参数如下：

--model：onnx模型加载路径。
--dataset：数据集所在路径。
--device：实现代码的设备。可选值为"gcu"、"cpu"或"gpu"。
--batchsize：batch size。
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- GCU环境运行

```bash
# FP16混合精度

python3 run_onnx.py --model model/gpt2_small-huggingface-op13-fp32-seqN-nocache.onnx --batchsize 1 --dataset data --device gcu
```

```bash
# FP32精度
export ORT_TOPSINFERENCE_FP16_ENABLE=0
 
python3 run_onnx.py --model model/gpt2_small-huggingface-op13-fp32-seqN-nocache.onnx --batchsize 1 --dataset data --device gpu
```

- CPU环境运行

```bash
python3 run_onnx.py --model model/gpt2_small-huggingface-op13-fp32-seqN-nocache.onnx --batchsize 1 --dataset data --device cpu
```

可通过屏幕打印查看结果。测试数据集的测试结果打印格式如下：

```bash
 Final report:
 {
    "model": "<path/to/onnx>",
    "dataset": "<path/to/dataset>"
    "device": "gcu",
    "perplexity": xxxxx,
 }
```

## 模型描述

### 性能

#### 评估性能

| 参数 | GCU <br>（FP16混合精度）| GCU <br>（FP32） | CPU <br>（FP32）|
| :--------------------------: | :--------------------------: | :--------------------------: | :--------------------------: |
| 资源 | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | X86_64 Intel CPU, 2.10GHZ, 内存 32G; 系统 Linux|
| 上传日期 | 2023-03-20 | 2023-03-20 | 2023-03-20 |
| onnxruntime-topsinference版本 | 1.9.1 | 1.9.1 | 1.9.1 |
| 数据集 | wikitext | wikitext | wikitext |
| 测试精度 | FP16混合精度 | FP32 | FP32 |
| perplexity | 29.383 | 29.372 | 29.372 |

## 随机情况说明

暂无
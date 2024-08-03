## README BERT

# 目录

<!-- TOC -->

- [目录](#目录)
    - [Bert描述](#bert描述)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [模型文件](#模型文件)
    - [环境要求](#环境要求)
    - [快速入门](#快速入门)
        - [设置PYTHONPATH](#设置pythonpath)
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

## Bert描述

Bert是一种预训练语言表征的方法，意味着我们在大型文本语料库（如维基百科）上训练一个通用的 "语言理解 "模型，然后将该模型用于我们关心的下游NLP任务（如问题回答）。BERT优于以前的方法，因为它是第一个用于预训练NLP的无监督、深度双向的系统。

[论文](https://arxiv.org/pdf/1810.04805v2.pdf)： Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *North American Chapter Of The Association For Computational Linguistics*. 2019.

## 模型架构

Bert base由LayerNorm层、Multi-head Attention和MLP组成。L（网络层数）=12, H（隐藏层维度）=768, A（Attention 多头个数）=12, Total Parameters= 12*768*12=110M。Bert可以适应许多类型的NLP任务: 句子级别（例如，SST-2）、句子对级别（例如，MultiNLI）、单词级别（例如，NER）和文本阅读（例如，SQuAD）。注意，本推理中使用的测试任务就是文本阅读。

## 数据集

使用的数据集：[SQuAD 1.1](<https://data.deepai.org/squad1.1.zip>)

- 数据集大小：33.50M，约500+文章的10万+问答对
    - 训练集：28.88M
    - 测试集：4.62M
- 数据格式：json文件
    - 注意：本测试使用的是测试数据集[dev-v1.1.json](<https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json>)数据评测在[evaluate-v1.1.py](<https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py>)中处理。
- 下载数据集。目录结构如下：

```data
├─dev-v1.1.json
└─evaluate-v1.1.py
```
注意在测试中需要原模型的配置文件，下载后将需要的文件放到对应目录。点击此处下载[BERT-Base, Uncased](<https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip>)。

## 模型文件

- 下载bert模型放在./model中。

[bert_base-squad-nvidia-op13-fp32-N.onnx](<https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/bert/bert_base-squad-nvidia-op13-fp32-N.onnx?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1671181960;2535181960&q-key-time=1671181960;2535181960&q-header-list=&q-url-param-list=&q-signature=b55cbc7faf17139cb35742fed841e1cbbf563ff9>)


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

### 设置PYTHONPATH

- 您需要将common文件夹的上级目录加入到PYTHONPATH环境变量

```shell
export PYTHONPATH=<parent/path/of/common>
```

也就是让PYTHONPATH路径呈现如下结构

```shell
PYTHONPATH
    ├── common
    └── nlp/bert
            └── run_onnx.py
```

### 运行

```python
# 进入脚本目录，使用bert base onnx推理
bash run.sh [ONNX_PATH] [CONFIG_PATH] [DATA_PATH] [ONNXC_VERSION] [DEVICE]
# example: bash run.sh ./model/bert_base-squad-nvidia-op13-fp32-N.onnx ./config ./data nvidia gcu

```

## 脚本说明

### 脚本及样例代码

```shell
./
├── mdoel
│     └── bert_base-squad-nvidia-op13-fp32-N.onnx 
│ 
├── config
│     ├── bert_config.json
│     └── vocab.txt
│ 
├── data
│      ├── dev-v1.1.json
│      └── evaluate-v1.1.py
│
├──requirements.txt
│
├──create_squad_data.py
│
├──run.sh
│
├──run_onnx.py
│
├──tokenization.py
│
└──README.md
```

### 脚本参数

```python
run_onnx.py中主要参数如下：

--vocab_file：词典对应路径。
--bert_config_file：配置文件路径。
--predict_batch_size：推理使用数据批次大小。
--onnx_path：onnx模型加载路径。
--model_ver：onnx模型来源版本，主要用来设置输入名称和形状等。
--device：实现代码的设备。可选值为"GCU"、"CPU"或"GPU"。
--predict_file：测试数据集路径。
--eval_script：数据集评估脚本所在路径
```

### 评估过程

#### 评估

在运行以下命令之前，请检查用于评估的检查点路径。

- GCU环境运行

```bash
 # FP16混合精度

 bash run.sh ./model/bert_base-squad-nvidia-op13-fp32-N.onnx ./config ./data nvidia gcu
```

```bash
 # FP32精度推理
 export ORT_TOPSINFERENCE_FP16_ENABLE=0
 
 bash run.sh ./model/bert_base-squad-nvidia-op13-fp32-N.onnx ./config ./data nvidia gcu
```

- GPU环境运行

```bash
 bash run.sh ./model/bert_base-squad-nvidia-op13-fp32-N.onnx ./config ./data nvidia gpu
```

- CPU环境运行

```bash
 bash run.sh ./model/bert_base-squad-nvidia-op13-fp32-N.onnx ./config ./data nvidia cpu
```

可通过屏幕打印查看结果。测试数据集的测试结果打印格式如下：

```bash
 Final report:
 {
    "model": "<path/to/onnx>",
    "dataset": "<path/to/dataset>"
    "device": "gcu",
    "exact_match": 0.xxxxx,
    "f1": 0.xxxxx,
 }
```



## 模型描述

### 模型精度

| 参数 | GCU <br>（FP16混合精度）| GCU <br>（FP32） | GPU <br>（FP32）| CPU <br>（FP32）|
| :--------------------------: | :--------------------------: | :--------------------------: | :--------------------------: | :--------------------------: |
| 资源 | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | Enflame 云燧i20 2.60GHz, 内存 16G; 系统 Linux | Nvidia A10 Tensor Core GPU, 24GB GDDR6显存; 系统 Linux | X86_64 Intel CPU, 2.10GHZ, 内存 32G; 系统 Linux|
| 上传日期 | 2022-12-02 | 2022-12-02 | 2022-12-02 | 2022-12-02 |
| onnxruntime-topsinference版本 | 1.9.1 | 1.9.1 | 1.9.1 | 1.9.1 |
| 数据集 | SQuAD 1.1 | SQuAD 1.1 | SQuAD 1.1 | SQuAD 1.1 |
| 测试精度 | FP16混合精度 | FP32 | FP32 | FP32 |
| exact_match | 81.47588| 81.53264 | 81.53264| 81.53264|
| f1 | 88.62022 | 88.64774 | 88.64774 | 88.64774

## 随机情况说明

暂无

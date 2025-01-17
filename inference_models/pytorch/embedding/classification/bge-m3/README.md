# README BGE-M3

# 目录

<!-- TOC -->
- [目录](#目录)
  - [bge-m3介绍](#bge-m3介绍)
  - [数据集](#数据集)
  - [模型文件](#模型文件)
  - [环境要求](#环境要求)
  - [模型验证运行示例](#模型验证运行示例)
    - [批量离线推理](#功能验证)
    - [性能测试](#性能测试)
  - [模型验证结果示例](#模型验证结果示例)
    - [批量离线推理结果示例](#功能验证结果示例)
    - [性能测试结果示例](#性能测试结果示例)

<!-- /TOC -->

## bge-m3介绍

bge-m3模型是一种新型的嵌入模型，由北京智源人工智能研究院开发，以其多功能性、多语言性和多粒度性而著称。在多语言性方面，BGE-M3能够支持超过100种语言，达到多语言和跨语言检索任务的新状态。多功能性方面，bge-m3能够同时执行嵌入模型的三种常见检索功能：密集检索、多向量检索和稀疏检索，为实际信息检索应用提供了统一的模型基础。在多粒度性方面，bge-m3能够处理从短句到长文档（最长可达8192个标记）的不同粒度的输入。bge-m3提出了一种新颖的自我知识蒸馏方法，使用来自不同检索功能的相关性评分增强训练质量。通过优化批处理策略，bge-m3实现了大批量和高训练吞吐量，从而确保嵌入的区分度。

[论文：Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.](https://arxiv.org/abs/2402.03216)

## 数据集

## 模型文件

- 模型名称：bge-m3
- 模型下载链接：
  - url：https://huggingface.co/BAAI/bge-m3
  - commit id：5617a9f61b028005a4858fdac845db406aefb181
  - branch：main
- 下载后放入您指定的存储目录

## 环境要求

- 硬件（GCU）
  - 准备GCU处理器搭建硬件环境
- 请完成`TopsRider`软件栈安装，安装过程请参考《TopsRider软件栈安装手册》
- TopsRider：3.2.204
- 通过以下命令安装依赖

```bash
  python3 -m pip install -r requirements.txt
```

## 模型验证运行示例

您可以按照如下步骤进行评估：

### 功能验证

```shell
python3 demo_bge.py \
 --device gcu \
 --model_dir MODEL_DIR \
 --model_type bi-encoder \
 --dtype float16
```

- 测试脚本主要参数如下：
  - device:模型推理使用的设备
  - model_dir:模型所在的目录
  - model_type:模型类型，使用bge-3m模型时，请设置为bi-encoder
  - dtype:模型的数据类型，为float16或者float32

### 性能测试

```shell
python3 benchmark.py 、
 --device gcu \
 --model_dir MODEL_DIR \
 --model_type bi-encoder \
 --dtype float16 \
 --max_len 128 \
 --bs 1
```

- 测试脚本主要参数如下：
  - device:模型推理使用的设备
  - model_dir:模型所在的目录
  - model_type:模型类型，使用bge-3m模型时，请设置为bi-encoder
  - dtype:模型的数据类型，为float16或者float32
  - max_len:限制模型输入的最大句子长度
  - bs:batch_size

## 模型验证结果示例

- 模型输出参考（性能数据以实测为主）：

### 功能验证结果示例

```
Sentence: This framework generates embeddings for each input sentence
Embedding: [ ... ]
Sentence: Sentences are passed as a list of strings.
Embedding: [ ... ]
Sentence: The quick brown fox jumps over the lazy dog.
Embedding: [ ... ]
Sentence: 样例文档-1
Embedding: [ ... ]
```

### 性能测试结果示例

```
model xxx
max length xxx
batch size xxx
dtype xxx
average time per step(s) xxx
```
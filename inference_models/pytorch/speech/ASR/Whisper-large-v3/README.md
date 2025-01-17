# README WHISPER-LARGE-V3

# 目录

<!-- TOC -->
- [目录](#目录)
  - [whisper-large-v3介绍](#whisper-large-v3介绍)
  - [数据集](#数据集)
  - [模型文件](#模型文件)
  - [环境要求](#环境要求)
  - [模型验证运行示例](#模型验证运行示例)
  - [模型验证结果示例](#模型验证结果示例)

<!-- /TOC -->

## whisper-large-v3介绍

whisper-large-v3是OpenAI于2022年12月发布的语音处理系统。虽然论文名字是Robust Speech Recognition via Large-Scale Weak Supervision，但不只是具有语音识别能力，还具备语音活性检测（VAD）、声纹识别、语音翻译（其他语种语音到英语的翻译）等能力。whisper-large-v3采用了更多的数据，达到了惊人的500万小时（一年=8765.81277 小时），其中100万小时是弱标签，400万小时是v2生成的。whisper-large-v3 相比 whisper-large-v2 在各个语言上有10%-20%的效果提升。

[论文：Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever. Robust Speech Recognition via Large-Scale Weak Supervision.](https://arxiv.org/abs/2212.04356)

## 数据集

- 数据集名称：Librispeech
- 数据集准备:
  - 下载：https://www.openslr.org/resources/12/dev-clean.tar.gz
  - 解压
  - 执行以下脚本，生成 dev_clean_test.txt， 请保证dev_clean_test.txt和LibriSpeech在同一级目录
  ```
    python process_libspeech.py --data_path ./LibriSpeech/ --save_path ./
  ```

## 模型文件

- 模型名称：whisper-large-v3
- 模型下载链接：
  - url：https://huggingface.co/openai/whisper-large-v3
  - commit id：1ecca609f9a5ae2cd97a576a9725bc714c022a93
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

通过官方网站安装Onnxruntime-TopsInference后，您可以按照如下步骤进行评估：

```shell
python3 test_whisper_large_v3.py \
 --model_path MODEL_PATH \
 --data_path DATA_PATH  \
 --batch_size 16 \
 --device gcu
```

- 测试脚本主要参数如下：
  - model_path: whisper-large-v3模型文件路径
  - data_path: 测试数据集路径，dev_clean_test.txt和LibriSpeech在同一级目录
  - device: 模型推理使用的设备
  - batch_size: 模型推理的batch size

## 模型验证结果示例

- 模型输出参考（性能数据以实测为主）：

```
Final report:
{
    "python": "xxx",
    "tops-inference": "xxx",
    "onnxruntime": "xxx",
    "model": "whisper_large_v3",
    "batch_size": xxx,
    "device": "gcu",
    "total_time": xxx,
    "total_samples": xxx,
    "average latency": xxx,
    "throughput": xxx,
    "accuracy": xxx
}

```
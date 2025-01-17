## README Meta-Llama-3.1-8B-Instruct

# 目录

<!-- TOC -->

- [目录](#目录)
  - [Meta-Llama-3.1-8B-Instruct介绍](#meta-llama-31-8b-instruct介绍)
  - [模型文件](#模型文件)
  - [环境要求](#环境要求)
  - [模型验证运行示例](#模型验证运行示例)
    - [批量离线推理](#批量离线推理)
    - [性能测试](#性能测试)
  - [模型验证结果示例](#模型验证结果示例)
    - [批量离线推理结果示例](#批量离线推理结果示例)
    - [性能测试结果示例](#性能测试结果示例)

<!-- /TOC -->

## Meta-Llama-3.1-8B-Instruct介绍

Meta-Llama-3.1-8B-Instruct是Meta开发的Llama3.1系列语言大模型的一部分，该系列还包括其他如70B和405B模型。Llama3.1系列的特点包括多语言支持、长达 128K 的上下文长度、先进的工具使用能力和增强的推理能力。这些功能为高级应用，如长文本摘要、多语言对话代理和编程助手等提供了支持。Meta-Llama-3.1-8B-Instruct是Meta-Llama-3.1-8B的微调版本，在对话任务上会表现得更为出色。该模型可支持的最大上下文长度为128K。vLLM已支持该模型的推理。本文档介绍在Enflame GCU上基于vLLM进行Meta-Llama-3.1-8B-Instruct的推理及性能评估过程。

[论文：Llama Team, AI @ Meta. The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)

## 模型文件

模型权重可以通过以下链接下载：
- 使用HuggingFace下载：
    - [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main)
    - 分支：main
    - Meta-Llama-3.1-8B-Instruct的HuggingFace下载需要经过模型所有者的许可，您可通过HuggingFace申请许可
- 或者使用ModelScope下载：
    - [Meta-Llama-3.1-8B-Instruct](https://modelscope.cn/models/llm-research/meta-llama-3.1-8b-instruct/files)
    - 分支：master
    - commit id：d02f94ee

下载后将所有文件放入`Meta-Llama-3.1-8B-Instruct`文件夹

## 环境要求

软硬件需求
- OS：ubuntu 22.04
- Python：3.8 - 3.10
- 加速卡：燧原S60
- TopsRider：3.2.204

推理框架安装
- Meta-Llama-3.1-8B-Instruct基于`vLLM`推理框架进行评估测试
- 以下步骤基于拟使用的 `Python3`版本, 请先安装对应`Python3`版本的所需依赖，需要在**docker**内安装：
- 安装`vLLM`之前，请完成`TopsRider`软件栈安装，安装过程请参考《TopsRider软件栈安装手册》;
- 首先通过如下命令检查`vllm`及相关依赖是否已经安装：
    ```shell
    python3 -m pip list | grep vllm
    python3 -m pip list | grep xformers
    python3 -m pip list | grep tops-extension
    ```
- 如果已经正常安装，可以显示如下内容：
    ```
    vllm                              <version>+gcu
    xformers                          <version>
    tops-extension                    <version>
    ```
- 如果未安装，可以通过`TopsRider`完成`vllm`安装：
    ```shell
    ./Topsrider_xxx.run -y -C vllm
    ```

依赖安装

- 进入 Meta-Llama-3.1-8B-Instruct 根目录，执行：

  ```bash
  pip3 install -r requirements.txt
  ```

## 模型验证运行示例

您可以使用如下命令运行vllm_utils模块中的相关脚本，对模型进行测试

### 批量离线推理

示例运行命令

```bash
python3 -m vllm_utils.benchmark_test \
 --model=[path_of_Meta-Llama-3.1-8B-Instruct] \
 --tensor-parallel-size=1 \
 --device gcu \
 --demo=te \
 --dtype=bfloat16 \
 --output-len=256 \
 --max-model-len=32768
```

参数说明

```text
使用vllm_utils.benchmark_test进行批量离线推理使用的主要参数如下：

--model：模型加载路径。
--tensor-parallel-size：推理时使用的gpu数量。
--device：进行推理的设备类型。可选值为"gcu"，"cpu"或"cuda"。
--demo：推理使用的示例输入，以下是可选值及其代表的示例输入类型：
    "te": "text-english"
    "tc": "text-chinese"
    "ch": "chat"
    "chc": "character-chat"
    "cc": "code-completion"
    "ci": "code-infilling"
    "cin": "code-instruction"
--dtype：模型权重以及激活层的数据类型
--output-len：每个prompt得到的输出长度
--max-model-len：模型最大的文本序列长度（输入与输出均计入）
```
注：
- Meta-Llama-3.1-8B-Instruct支持的`max-model-len`为最大131072，单张S60可支持32768

### 性能测试

示例运行命令

```bash
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path_of_Meta-Llama-3.1-8B-Instruct] \
 --tensor-parallel-size=1 \
 --device gcu \
 --max-model-len=32768 \
 --tokenizer=[path_of_Meta-Llama-3.1-8B-Instruct] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=8 \
 --block-size=64 \
 --dtype=bfloat16
```

参数说明

```text
使用vllm_utils.benchmark_test进行性能测试使用的主要参数如下：

--perf：使用性能评估模式。
--model：模型加载路径。
--tensor-parallel-size：推理时使用的gpu数量。
--device：进行推理的设备类型。可选值为"gcu"，"cpu"或"cuda"。
--max-model-len：模型最大的文本序列长度（输入与输出均计入）
--tokenizer：模型tokenizer加载路径，一般与模型路径一致
--input-len：输入的prompt长度
--output-len：每个prompt得到的输出长度
--num-prompts：输入的prompt数量
--block-size：Paged Attention的block大小
--dtype：模型权重以及激活层的数据类型
```
注：
- Meta-Llama-3.1-8B-Instruct支持的`max-model-len`为最大131072，单张S60可支持32768
- `input-len`、`output-len`和`num-prompts`可按需调整
- 配置`output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency

## 模型验证结果示例

### 批量离线推理结果示例

以--output-len=32为例：
```text
Prompt: 'Hello, my name is', Generated text: ' Emily and I am a 25-year-old freelance writer and editor. I have a passion for storytelling and a knack for crafting compelling narratives. I have been writing'
Prompt: 'The president of the United States is', Generated text: ' the head of state and head of government of the United States. The president serves a four-year term and is limited to two terms. The president is elected through'
Prompt: 'The capital of France is', Generated text: ' a city of romance, art, fashion, and cuisine. Paris is a must-visit destination for anyone who loves history, architecture, and culture. From the'
Prompt: 'The future of AI is', Generated text: ' bright, but it also raises concerns about bias, accountability, and the impact on jobs. Here are some of the key challenges and opportunities that AI will face in'
```

### 性能测试结果示例

推理结束后会输出性能指标，自动存储在当前目录中生成的.csv文件内

```text
***Perf Info***
{
    "latency_num_prompts": "xxx ms",
    "latency_per_token": "xxx ms",
    "request_per_second": "xxx requests/s",
    "token_per_second": "xxx tokens/s",
    "prefill_latency_per_token": "xxx ms",
    "decode_latency_per_token": "xxx ms",
    "decode_throughput": "xxx tokens/s"
}
```
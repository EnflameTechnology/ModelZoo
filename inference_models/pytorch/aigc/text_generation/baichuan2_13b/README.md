## README baichuan2_13b

# 目录

<!-- TOC -->

- [目录](#目录)
  - [baichuan2\_13b介绍](#baichuan2_13b介绍)
  - [模型文件](#模型文件)
  - [环境要求](#环境要求)
  - [模型验证运行示例](#模型验证运行示例)
    - [批量离线推理](#批量离线推理)
    - [性能测试](#性能测试)
  - [模型验证结果示例](#模型验证结果示例)
    - [批量离线推理结果示例](#批量离线推理结果示例)
    - [性能测试结果示例](#性能测试结果示例)

<!-- /TOC -->

## baichuan2_13b介绍



baichuan2_13b是一个文生文开源双语大模型，由百川智能研发，采用 2.6 万亿 Tokens 的高质量语料训练，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。该模型可支持的最大上下文长度为4K，vLLM已支持该模型的推理。本文档介绍在Enflame GCU上基于vLLM进行baichuan2_13b的推理及性能评估过程。

[论文：Baichuan Inc. Baichuan 2: Open Large-scale Language Models.](https://arxiv.org/abs/2309.10305)

## 模型文件

模型权重可以通过以下链接下载：
- 使用HuggingFace下载：
    - [baichuan2_13b](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/tree/main)
    - 分支：main
    - commit id：c6f590cab590cf33e78ad834dbd5f9bd6df34a94
- 或者使用ModelScope下载：
    - [baichuan2_13b](https://www.modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base/files)
    - 分支：master
    - commit id：174d87a40823cb1f9cf4bc881b77ca9b4a396792

下载后将所有文件放入`baichuan2_13b`文件夹

## 环境要求

软硬件需求
- OS：ubuntu 22.04
- Python：3.10
- 加速卡：燧原S60
- TopsRider：3.2.204

推理框架安装
- baichuan2_13b基于`vLLM`推理框架进行评估测试
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

## 模型验证运行示例

您可以使用如下命令运行vllm_utils模块中的相关脚本，对模型进行测试

### 批量离线推理

示例运行命令

```bash
python3 -m vllm_utils.benchmark_test \
 --model=[path_of_baichuan2_13b] \
 --tensor-parallel-size=1 \
 --device gcu \
 --demo=te \
 --dtype=float16 \
 --output-len=256
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
```

### 性能测试

示例运行命令

```bash
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path_of_baichuan2_13b] \
 --tensor-parallel-size=1 \
 --device gcu \
 --max-model-len=4096 \
 --tokenizer=[path_of_baichuan2_13b] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=64 \
 --block-size=64 \
 --dtype=float16
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
- baichuan2_13b支持的`max-model-len`为最大4096
- `input-len`、`output-len`和`num-prompts`可按需调整
- 配置`output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency

## 模型验证结果示例

### 批量离线推理结果示例

以--output-len=32为例：
```text
Prompt: 'Hello, my name is', Generated text: ' [Your Name] and I am a [Your Age] year old [Gender] from [Your Country]. I am currently studying [Your Course] at'
Prompt: 'The president of the United States is', Generated text: ' the head of state and head of government of the United States of America. The president directs the executive branch of the federal government and is the commander-in-'
Prompt: 'The capital of France is', Generated text: ' Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of the United Kingdom is London.\nThe capital of'
Prompt: 'The future of AI is', Generated text: ' in the hands of the people who are creating it. As AI continues to evolve, it is crucial for developers, researchers, and policymakers to consider the ethical implications'
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
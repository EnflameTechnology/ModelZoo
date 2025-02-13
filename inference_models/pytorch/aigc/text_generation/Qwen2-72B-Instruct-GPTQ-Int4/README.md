## README Qwen2-72B-Instruct-GPTQ-Int4

# 目录

<!-- TOC -->

- [目录](#目录)
  - [Qwen2-72B-Instruct-GPTQ-Int4介绍](#Qwen2-72B-Instruct-GPTQ-Int4-instruct介绍)
  - [模型文件](#模型文件)
  - [环境要求](#环境要求)
  - [模型验证运行示例](#模型验证运行示例)
    - [批量离线推理](#批量离线推理)
    - [性能测试](#性能测试)
  - [模型验证结果示例](#模型验证结果示例)
    - [批量离线推理结果示例](#批量离线推理结果示例)
    - [性能测试结果示例](#性能测试结果示例)

<!-- /TOC -->

## Qwen2-72B-Instruct-GPTQ-Int4介绍

Qwen2-72B-Instruct 是阿里巴巴通义千问团队开发的新一代大型语言模型，它是 Qwen2 系列中参数规模最大的模型。这个模型基于 Transformer 架构，具有 SwiGLU 激活、注意力 QKV bias、群组查询注意力等先进技术。Qwen2-72B-Instruct 在多个基准测试中表现出色，特别是在语言理解、语言生成、多语言能力、编码、数学、推理等方面。它在代码方面融入了 CodeQwen1.5 的成功经验，实现了在多种编程语言上的显著效果提升。在数学方面，大规模且高质量的数据帮助 Qwen2-72B-Instruct 实现了数学解题能力的飞升。

[论文：Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)

## 模型文件

模型权重可以通过以下链接下载：
- 使用HuggingFace下载：
    - [Qwen2-72B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-72B-Instruct-GPTQ-Int4)
    - 分支：main
    - Qwen2-72B-Instruct-GPTQ-Int4的HuggingFace下载需要经过模型所有者的许可，您可通过HuggingFace申请许可
- 或者使用ModelScope下载：
    - [Qwen2-72B-Instruct-GPTQ-Int4](https://www.modelscope.cn/models/Qwen/Qwen2-72B-Instruct-GPTQ-Int4)
    - 分支：master
    - commit id：c7e75f6b

下载后将所有文件放入`Qwen2-72B-Instruct-GPTQ-Int4`文件夹

## 环境要求

软硬件需求
- OS：ubuntu 22.04
- Python：3.8 - 3.10
- 加速卡：燧原S60
- TopsRider：3.2.204

推理框架安装
- Qwen2-72B-Instruct-GPTQ-Int4基于`vLLM`推理框架进行评估测试
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
 --model=[path of Qwen2-72B-Instruct-GPTQ-Int4] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --output-len=512 \
 --demo=te \
 --dtype=float16 \
 --device=gcu \
 --quantization=gptq
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
- Qwen2-72B-Instruct-GPTQ-Int4支持的`max-model-len`为最大32768

### 性能测试

示例运行命令

```bash
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen2-72B-Instruct-GPTQ-Int4] \
 --max-model-len=32768 \
 --tokenizer=[path of Qwen2-72B-Instruct-GPTQ-Int4] \
 --input-len=1024 \
 --output-len=500 \
 --num-prompts=1 \
 --tensor-parallel-size=4 \
 --block-size=64 \
 --quantization=gptq \
 --device=gcu
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
- Qwen2-72B-Instruct-GPTQ-Int4支持的`max-model-len`为最大32768;
- `input-len`、`output-len`和`num-prompts`可按需调整
- 配置`output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency

## 模型验证结果示例

### 批量离线推理结果示例

以--output-len=32为例：
```text
Prompt: 'Hello, my name is', Generated text: ' Dr. David B. Samadi. I am a board certified urologist, the chairman of urology and the chief of robotic surgery at Len'
Prompt: 'The president of the United States is', Generated text: ' the head of the executive branch of the U.S. government and is the commander-in-chief of the United States Armed Forces. The president is also the chief diplomat'
Prompt: 'The capital of France is', Generated text: ' Paris. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital of Italy is Rome. The capital of Greece is Athens. The capital'
Prompt: 'The future of AI is', Generated text: ' here, and it’s not just about robots and self-driving cars. AI is transforming the way we live, work, and play, and it’s only going'
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
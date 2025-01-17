## README Qwen2-7B

# 目录

<!-- TOC -->

- [目录](#目录)
  - [Qwen2-7B介绍](#Qwen2-7B-instruct介绍)
  - [模型文件](#模型文件)
  - [环境要求](#环境要求)
  - [模型验证运行示例](#模型验证运行示例)
    - [批量离线推理](#批量离线推理)
    - [性能测试](#性能测试)
  - [模型验证结果示例](#模型验证结果示例)
    - [批量离线推理结果示例](#批量离线推理结果示例)
    - [性能测试结果示例](#性能测试结果示例)

<!-- /TOC -->

## Qwen2-7B介绍

Qwen2-7B 是阿里巴巴通义千问团队发布的 Qwen2 系列开源模型之一，该系列模型包括不同尺寸的预训练和指令微调模型，如 Qwen2-0.5B、Qwen2-1.5B、Qwen2-7B、Qwen2-57B-A14B 以及 Qwen2-72B。Qwen2-7B 模型在多个评测上表现出色，尤其在代码及中文理解上，相比同等规模的其他开源模型如 Llama3-8B、GLM4-9B 等，Qwen2-7B-Instruct 指令微调的中等尺寸模型依然能取得显著的优势。Qwen2-7B 模型基于 Transformer 架构，使用下一个词预测进行训练。它支持长达 131,072 个 token 的上下文长度，能够处理大量输入文本
。此外，Qwen2-7B 模型在长上下文能力、多语言评估以及安全性和责任方面都进行了优化和提升

[论文：Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)

## 模型文件

模型权重可以通过以下链接下载：
- 使用HuggingFace下载：
    - [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)
    - 分支：main
    - Qwen2-7B的HuggingFace下载需要经过模型所有者的许可，您可通过HuggingFace申请许可
- 或者使用ModelScope下载：
    - [Qwen2-7B](https://www.modelscope.cn/models/Qwen/Qwen2-7B)
    - 分支：master
    - commit id：417d39e7

下载后将所有文件放入`Qwen2-7B`文件夹

## 环境要求

软硬件需求
- OS：ubuntu 22.04
- Python：3.8 - 3.10
- 加速卡：燧原S60
- TopsRider：3.2.204

推理框架安装
- Qwen2-7B基于`vLLM`推理框架进行评估测试
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
    --demo='te' \
    --model= [path of Qwen2-7B] \
    --tokenizer= [path of Qwen2-7B] \
    --num-prompts 1 \
    --max-model-len=32768 \
    --block-size=64 \
    --output-len=256 \
    --device=gcu \
    --dtype=float16 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.945
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
- Qwen2-7B支持的`max-model-len`为最大131072，单张S60可支持32768

### 性能测试

示例运行命令

```bash
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model [path of Qwen2-7B] \
    --tensor-parallel-size 1 \
    --max-model-len=32768 \
    --input-len=8000 \
    --output-len=8000 \
    --dtype=float16 \
    --device gcu \
    --num-prompts=1 \
    --block-size=64 \
    --gpu-memory-utilization=0.945
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
- Qwen2-7B支持的`max-model-len`为最大131072，单张S60可支持32768
- `input-len`、`output-len`和`num-prompts`可按需调整
- 配置`output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency

## 模型验证结果示例

### 批量离线推理结果示例

以--output-len=32为例：
```text
Prompt: 'Hello, my name is', Generated text: ' Dr. David M. Berman. I am a board-certified plastic surgeon with over 20 years of experience. I am a member of the American'
Prompt: 'The president of the United States is', Generated text: ' the head of state and head of government of the United States. The president directs the executive branch of the federal government and is the commander-in-chief of the United'
Prompt: 'The capital of France is', Generated text: ' Paris. It is the largest city in France and is located in the north-central part of the country. Paris is a major international center for business, fashion,'
Prompt: 'The future of AI is', Generated text: ' here. It’s not just a buzzword anymore. It’s a reality that’s changing the way we live, work, and interact with the world around us'
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
## README llava-1.5-7b-hf

# 目录

<!-- TOC -->

- [目录](#目录)
  - [llava-1.5-7b-hf介绍](#llava-1.5-7b-hf介绍)
  - [模型文件](#模型文件)
  - [环境要求](#环境要求)
  - [模型验证运行示例](#模型验证运行示例)
    - [批量离线推理](#批量离线推理)
    - [性能测试](#性能测试)
  - [模型验证结果示例](#模型验证结果示例)
    - [批量离线推理结果示例](#批量离线推理结果示例)
    - [性能测试结果示例](#性能测试结果示例)

<!-- /TOC -->

## llava-1.5-7b-hf介绍

LLaVA模型是一种通过指令微调提升多模态领域能力的自回归大语言模型，在结构上使用了transformer架构。该模型首次尝试利用仅语言的GPT-4生成多模态的语言-图像指令数据，并在此类生成数据上进行微调，形成了LLaVA。LLaVA将视觉编码器和大型语言模型连接起来，实现了通用的视觉和语言理解。实验结果显示，LLaVA在多模态对话能力上表现出色，有时在处理未见过的图像和指令时，与多模态的GPT-4表现相似。此外，当在科学问答类任务上进行微调时，LLaVA与GPT-4的协同作用达到了92.53%的最新准确率。研究团队已将GPT-4生成的微调数据、模型和代码公开，供广大研究者使用。

[论文：Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee. LLaVA: Large Language and Vision Assistant Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

## 模型文件

模型权重可以通过以下链接下载：
- 使用HuggingFace下载：
    - [llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/main)
    - 分支：main
    - commit id: 8c85e9a4d626b7b908448be32c1ba5ad79b95e76

下载后将所有文件放入`llava-1.5-7b-hf`文件夹

## 环境要求

软硬件需求
- OS：ubuntu 22.04
- Python：3.8 - 3.10
- 加速卡：燧原S60
- TopsRider：3.2.204

推理框架安装
- llava-1.5-7b-hf基于`vLLM`推理框架进行评估测试
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
python3 -m vllm_utils.benchmark_vision_language --demo \
 --model-type=llava \
 --model=[path of llava-1.5-7b-hf] \
 --device=gcu \
 --input-image="./car-balloons-800x525.jpg" \
 --prompt="Describe the picture."
```

参数说明

```text
使用vllm_utils.benchmark_vision_language进行批量离线推理使用的主要参数如下：

--demo: 使用少量样本的示例推理模式
--model-type：多模态模型的类型。
--model：模型加载路径。
--device：进行推理的设备类型。可选值为"gcu"，"cpu"或"cuda"。
--input-image: 输入的图像路径。
--prompt: 输入的自然语言prompt。
```

### 性能测试

示例运行命令

```bash
python3 -m vllm_utils.benchmark_vision_language --perf \
 --model-type=llava \
 --model=[path of llava-1.5-7b-hf] \
 --tensor-parallel-size=1 \
 --device gcu \
 --max-model-len=4096 \
 --input-len=2048 \
 --output-len=2048 \
 --num-prompts 1 \
 --batch-size=1 \
 --block-size=64 \
 --dtype=float16
```

参数说明

```text
使用vllm_utils.benchmark_vision_language进行性能测试使用的主要参数如下：

--perf：使用性能评估模式。
--model-type：多模态模型的类型。
--model：模型加载路径。
--tensor-parallel-size：使用张量并行时使用的设备数量。
--device：进行推理的设备类型。可选值为"gcu"，"cpu"或"cuda"。
--max-model-len：模型最大的文本序列长度（输入与输出均计入）
--input-len：输入的prompt长度
--output-len：每个prompt得到的输出长度
--num-prompts：输入的prompt数量
--batch-size: 每个batch中的prompt数
--block-size：Paged Attention的block大小
--dtype：模型权重以及激活层的数据类型
```
注：
- llava-1.5-7b-hf支持的`max-model-len`为最大4096
- `input-len`、`output-len`、`num-prompts`和`batch-size`可按需调整
- 配置`output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency
- 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`

## 模型验证结果示例

### 批量离线推理结果示例


```text
Prompt: 
USER: <image>Describe the picture.
ASSISTANT:, Generated text:  The image features a small yellow and pink car parked on a street, surrounded by a large number of balloons. The balloons are of various sizes and colors, creating a vibrant and festive atmosphere. The car is positioned in the center of the scene, with the balloons surrounding it from all sides. The street appears to be a part of a city, with buildings visible in the background.
There are also two people in the scene, one standing closer to the left side of the image and the other near the right side. They seem to be enjoying the sight of the car
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
    "decode_latency_per_token": "xxx ms"
}
```
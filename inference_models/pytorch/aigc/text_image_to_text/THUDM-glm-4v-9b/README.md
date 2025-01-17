## README GLM-4V-9B

# 目录

<!-- TOC -->

- [目录](#目录)
  - [GLM-4V-9B介绍](#GLM-4V-9B-instruct介绍)
  - [模型文件](#模型文件)
  - [环境要求](#环境要求)
  - [模型验证运行示例](#模型验证运行示例)
    - [批量离线推理](#批量离线推理)
    - [性能测试](#性能测试)
  - [模型验证结果示例](#模型验证结果示例)
    - [批量离线推理结果示例](#批量离线推理结果示例)
    - [性能测试结果示例](#性能测试结果示例)

<!-- /TOC -->

## GLM-4V-9B介绍

GLM-4V-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源多模态版本。 GLM-4V-9B 具备 1120 * 1120 高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4V-9B 表现出超越 GPT-4-turbo-2024-04-09、Gemini 1.0 Pro、Qwen-VL-Max 和 Claude 3 Opus 的卓越性能。

[论文：GLM Technical Report](https://arxiv.org/abs/2406.12793)

## 模型文件

模型权重可以通过以下链接下载：
- 使用HuggingFace下载：
    - [GLM-4V-9B](https://huggingface.co/THUDM/glm-4v-9b)
    - 分支：main
    - GLM-4V-9B的HuggingFace下载需要经过模型所有者的许可，您可通过HuggingFace申请许可
- 或者使用ModelScope下载：
    - [GLM-4V-9B](https://www.modelscope.cn/models/ZhipuAI/glm-4v-9b)
    - 分支：master
    - commit id：7e7a344e

下载后将所有文件放入`GLM-4V-9B`文件夹

## 环境要求

软硬件需求
- OS：ubuntu 22.04
- Python：3.8 - 3.10
- 加速卡：燧原S60
- TopsRider：3.2.204

推理框架安装
- GLM-4V-9B基于`vLLM`推理框架进行评估测试
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
python3 -m vllm_utils.benchmark_vision_language \
 --demo \
 --model-type=glm-4v \
 --model=[path of glm-4v-9b] \
 --input-image Test.jpg \
 --tensor-parallel-size 1 \
 --max-model-len=8192 \
 --output-len=128 \
 --dtype=bfloat16 \
 --device gcu \
 --block-size=64 \
 --trust-remote-code \
 --prompt-template '[gMASK] <sop> <|user|> \n {} <|assistant|>'
```

参数说明

```text
使用vllm_utils.benchmark_test进行批量离线推理使用的主要参数如下：

--model-type: VL模型的类型
--model：模型加载路径。
--input-image: 输入图像的路径
--tensor-parallel-size：推理时使用的gpu数量。
--device：进行推理的设备类型。可选值为"gcu"，"cpu"或"cuda"。
--demo：表示测试输出
--dtype：模型权重以及激活层的数据类型
--output-len：每个prompt得到的输出长度
--max-model-len：模型最大的文本序列长度（输入与输出均计入）
--prompt-template: 对话模版，Python format字符串格式
```
注：
- GLM-4V-9B支持的`max-model-len`为最大8192，单张S60可支持8192

### 性能测试

示例运行命令

```bash
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model [path of GLM-4V-9B] \
    --tensor-parallel-size 1 \
    --max-model-len=8192 \
    --input-len=4096 \
    --output-len=4096 \
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
- GLM-4V-9B支持的`max-model-len`为最大8192，单张S60可支持8192
- `input-len`、`output-len`和`num-prompts`可按需调整
- 配置`output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency

## 模型验证结果示例

### 批量离线推理结果示例

以--output-len=256为例：
```text
Prompt: [gMASK] <sop> <|user|> \n <|begin_of_image|>
USER: What is the content of this image?
ASSISTANT: <|assistant|>, Generated text: The image depicts a serene beach scene at sunset. The sky is a gradient of warm colors, transitioning from a deep blue to a soft orange hue, suggesting the sun is either setting or rising. There are a few wispy clouds scattered across the sky.

On the left side of the image, there is a person in black athletic attire, running along the shoreline. The person is captured in mid-stride, with one leg lifted and the other planted on the wet sand, which reflects the light and gives the impression of movement. The runner's reflection can be seen on the wet sand, creating a symmetrical image.

In the background, there are two large rock formations protruding from the ocean, with one having a natural archway. The rocks are covered in green vegetation, indicating they are possibly sea stacks or coastal cliffs. The water around the rocks is calm, with small waves gently lapping at the shore.

The overall mood of the image is peaceful and dynamic, with the contrast between the motion of the runner and the stillness of the natural surroundings creating a sense of harmony. The reflection on the wet sand adds depth and a sense of tranquility to the scene.
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
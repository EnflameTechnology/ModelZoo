# HunyuanDiT-v1.2 推理指南

## 概述

本文档介绍在 Enflame GCU 上基于 pytorch native 进行 HunyuanDiT-v1.2 的 text2image 任务的推理及性能评估过程

## 模型说明
混元DiT，一个基于Diffusion transformer的文本到图像生成模型，此模型具有中英文细粒度理解能力。为了构建混元DiT，作者精心设计了Transformer结构、文本编码器和位置编码。并且作者构建了完整的数据管道，用于更新和评估数据，为模型优化迭代提供帮助。为了实现细粒度的文本理解，

## 环境配置

以下步骤基于 `Python3.10`, 请先安装所需依赖：

* 安装环境：安装过程请参考《TopsRider 软件栈安装手册》，请根据手册完成 TopsRider 软件栈安装
* TopsRider：3.2.204

- 安装 torch_gcu

  注意：安装 torch_gcu-2.3.0 会自动安装 torch 2.3.0

  ```bash
  # 需要使用 root 权限
  ./TopsRider_{filename}.run -y -C torch-gcu-2
  ```

- 安装 diffusers

  ```bash
  pip3 install diffusers==0.30.2
  ```

- 进入 hydit_v1_2_diffusers 根目录，执行：

  ```bash
  pip3 install -r requirements.txt
  ```

### 准备模型

* 下载预训练模型：

  请从 [HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers/tree/main) 路径下下载全部内容到模型存放目录，以下用 `path_to_model_dir` 表示其路径
  - branch: `main`
  - commit id: `5e96094e0ad19e7f475de8711f03634ca0ccc40c`

## HunyuanDiT-v1.2 推理

### 执行推理

使用 HunyuanDiT-v1.2-Diffusers 进行推理，进入 HunyuanDiT-v1.2-Diffusers 目录，执行以下命令：

``` bash
python3 demo_hydit_v1_2_diffusers.py \
  --model_dir ${path_to_model_dir}  \
  --device gcu \
  --prompt '太阳微微升起，花园里的玫瑰花瓣上露珠晶莹剔透，一只瓢虫正在爬向露珠，背景是清晨的花园，微距镜头'  \
  --image_height 1024  \
  --image_width 1024  \
  --denoising_steps 30  \
  --output_dir results_hydit_v1_2_diffusers
```

其中，

* `--model_dir`: HunyuanDiT-v1.2 的预训练模型所在的目录
* `--image_height`: 生成图片的高度
* `--image_width`: 生成图片的宽度
* `--denoising_steps`: 去噪步数
* `--output_dir`: 保存生成图片的路径

其它参数及其含义请使用以下命令查看：

``` bash
python3 demo_hydit_v1_2_diffusers.py -h
```

### 性能评估

使用 HunyuanDiT-v1.2-Diffusers 进行推理：

``` bash
python3  benchmark_test_hydit_v1_2_diffusers.py \
 --model_dir ${path_to_model_dir}  \
 --device gcu \
 --prompt '太阳微微升起，花园里的玫瑰花瓣上露珠晶莹剔透，一只瓢虫正在爬向露珠，背景是清晨的花园，微距镜头'  \
 --image_height 1024  \
 --image_width 1024  \
 --denoising_steps 30  \
 --output_dir results_hydit_v1_2_diffusers \
 --warmup_count 3 \
 --eval_count 5
```

其中，

* `--model_dir`: HunyuanDiT 的预训练模型所在的目录
* `--image_height`: 生成图片的高度
* `--image_width`: 生成图片的宽度
* `--denoising_steps`: 去噪步数
* `--output_dir`: 生成图片以及性能报告的保存路径
* `--warmup_count`: warmup 次数
* `--eval_count`: 重复推理次数

其它参数及其含义请使用以下命令查看：

``` bash
python3  benchmark_test_hydit_v1_2_diffusers.py -h
``` 
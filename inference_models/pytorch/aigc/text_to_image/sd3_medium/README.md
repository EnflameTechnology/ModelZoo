# SD3_Medium 模型推理指南

## 概述

本文档介绍在 Enflame GCU 上基于 pytorch native 进行 stable-diffusion-3-medium 的 text2image 任务的推理及性能评估过程

## 模型说明

Stable Diffusion 3 Medium 是一种多模态扩散变换器（MMDiT）文本到图像模型，它在图像质量、排版、复杂提示理解和资源效率方面都具有显著提升的性能。Stable Diffusion 3 Medium使用三种不同的文本嵌入器（两种 CLIP 模型和 T5）来对文本表示进行编码，并使用一种改进的自动编码模型来对图像标记进行编码。

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
  
- 安装依赖

  进入 stable-diffusion-3-medium 根目录，执行：

  ```bash
  pip3 install -r requirements.txt
  ```

## 准备模型

- 下载预训练模型：

  请从 [stable-diffusion-3-medium-diffusers](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-3-medium-diffusers/files) 路径下下载全部内容到模型存放目录，以下用

    `path_to_model_dir` 表示其路径

    - branch: `master`
    - commit id: `3987867f`

## stable-diffusion-3-medium 推理

### 执行推理

使用 stable-diffusion-3-medium 进行推理，进入 stable-diffusion-3-medium 目录，执行以下命令：

``` bash
python3 demo_stable_diffusion_3_txt2img.py \
--model_dir ${path_to_model_dir} \
--num_images_per_prompt 1 \
--prompt 'photo of an astronaut on mars'  \
--prompt_2 'Portrait of gldot as a beautiful female model'  \
--prompt_3 'She is wearing a leather jacket, black jeans, dramatic lighting'  \
--negative_prompt 'monochrome' \
--negative_prompt_2 'poorly drawn hands' \
--negative_prompt_3 'ugly, blurry, flat chest' \
--seed 12345 \
--denoising_steps 30 \
--guidance_scale 7.5 \
--image_height 1024 \
--image_width 1024 \
--output_dir './results/sd3/txt2img/demo'  \
--device gcu
```

其中，

* `--model_dir`: stable-diffusion-3-medium-diffusers 的预训练模型所在的目录
* `--num_images_per_prompt`: 每组 prompt 生成图片的数量
* `--image_height`: 生成图片的高度
* `--image_width`: 生成图片的宽度
* `--prompt`: 正向提示词 1
* `--prompt_2`: 正向提示词 2
* `--prompt_3`: 正向提示词 3
* `--negative_prompt`: 反向提示词 1
* `--negative_prompt_2`: 反向提示词 2
* `--negative_prompt_3`: 反向提示词 3
* `--denoising_steps`: 去噪步数
* `--output_dir`: 保存生成图片的路径
* `--device`: 推理使用的设备，默认为'gcu'

其它参数及其含义请使用以下命令查看：

``` bash
python3 demo_stable_diffusion_3_txt2img.py -h
```

### 性能评估

进入 stable-diffusion-3-medium 目录，执行以下命令：

```bash
python3 benchmark_test_stable_diffuision_3_txt2img.py \
--model_dir ${path_to_model_dir} \
--num_images_per_prompt 1 \
--prompt 'photo of an astronaut on mars'  \
--prompt_2 'Portrait of gldot as a beautiful female model'  \
--prompt_3 'She is wearing a leather jacket, black jeans, dramatic lighting'  \
--negative_prompt 'monochrome' \
--negative_prompt_2 'poorly drawn hands' \
--negative_prompt_3 'ugly, blurry, flat chest' \
--seed 12345 \
--denoising_steps 30 \
--guidance_scale 7.5 \
--image_height 1024 \
--image_width 1024 \
--warmup_count 3 \
--eval_count 5 \
--output_dir './results/sd3/txt2img/benchmark'  \
--device gcu
```

其中，

* `--model_dir`: stable-diffusion-3-medium-diffusers 的预训练模型所在的目录
* `--num_images_per_prompt`: 每组 prompt 生成图片的数量
* `--image_height`: 生成图片的高度
* `--image_width`: 生成图片的宽度
* `--prompt`: 正向提示词 1
* `--prompt_2`: 正向提示词 2
* `--prompt_3`: 正向提示词 3
* `--negative_prompt`: 反向提示词 1
* `--negative_prompt_2`: 反向提示词 2
* `--negative_prompt_3`: 反向提示词 3
* `--denoising_steps`: 去噪步数
* `--output_dir`: 保存生成图片的路径
* `--warmup_count`: warmup次数
* `--eval_count`: 重复推理次数
* `--device`: 推理使用的设备，默认为'gcu'

其它参数及其含义请使用以下命令查看：

``` bash
python3 benchmark_test_stable_diffuision_3_txt2img.py -h
```

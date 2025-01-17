# Segment-Anything

## 模型说明
Segment Anything Model (SAM) 是一个设计精巧的架构，用于高效地生成物体掩膜。其核心组成部分是一个重量级的图像编码器，它能够输出图像嵌入。这一图像嵌入随后可以被不同类型的输入提示高效查询，以实时速度生成物体掩膜。SAM主要有3个部分组成：图像编码器、提示编码器和掩膜解码器。
![alt text](model_diagram.png)

* 图像来源：[Segment Anything](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#segment-anything)

## 环境依赖

* 安装环境：安装过程请参考《TopsRider软件栈安装手册》，TopsRider提供了torch-gcu.whl的安装包。完成TopsRider软件栈安装，即完成推理所需要基础环境的安装。
* TopsRider：3.2.204
* 如代码目录下存在requirements.txt文件，请按照requirements.txt中所示安装相应的依赖库

## 模型

- 预训练模型共3个模型，应放置在./models，可以从[segment-anything](https://github.com/facebookresearch/segment-anything)下载
   - vit_h: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
   - vit_l: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
   - vit_b: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## 数据准备

```shell
mkdir images && cd images
wget https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/notebooks/images/dog.jpg
wget https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/notebooks/images/groceries.jpg
wget https://raw.githubusercontent.com/facebookresearch/segment-anything/dca509fe793f601edb92606367a655c15ac00fdf/notebooks/images/truck.jpg
```

## automask测试

### 测试命令
```shell
python3 demo_automask.py \
--checkpoint [path of checkpoint] \
--device gcu \
--image_path ./images \
--save_path ./output
```

### 参数说明

- checkpoint: Segment Anything的权重文件
- device: 模型推理使用的设备，['cpu', 'cuda', 'gcu']
- image_path: 推理图像路径
- save_path: 推理结果保存路径

## prompt测试

### 测试命令

```shell
python3 demo_prompt.py \
--checkpoint [path of checkpoint] \
--device gcu \
--image_path ./images \
--ann_file ./annotation.json \
--save_path ./output
```

### 参数说明

- checkpoint: Segment Anything的权重文件
- device: 模型推理使用的设备，['cpu', 'cuda', 'gcu']
- image_path: 推理图像路径
- ann_file: 图像prompts文件
- save_path: 推理结果保存路径

## 模型验证结果示例

- 模型输出参考（性能数据以实测为主）：

### 功能验证结果示例

```
{
   "device": xxxx,
   "model": xxxx,
   "save_path": xxxx,
   "image_number": xxxx,
   "prompt_number": xxxx,
   "avg_infer_time_per_prompt(ms)": xxxx
}
```

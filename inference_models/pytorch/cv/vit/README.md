# ViT

## 模型说明
Vision Transformer（ViT）是由Google研究团队于 2020 年提出的一种基于自注意力机制的视觉模型。与传统的卷积神经网络不同，ViT 基于 Transformer 架构，可扩展性强，成为了 transformer 在 CV 领域应用的里程碑著作，也引爆了后续相关研究

## 环境依赖
* 安装环境：安装过程请参考《TopsRider软件栈安装手册》，TopsRider提供了torch-gcu.whl的安装包。完成TopsRider软件栈安装，即完成推理所需要基础环境的安装。
* TopsRider：3.2.204
* 如代码目录下存在requirements.txt文件，请按照requirements.txt中所示安装相应的依赖库

## 模型
- 模型名称：ViT
- [vit-large-patch16-384](https://huggingface.co/google/vit-large-patch16-384)
    - commit id：4b143e7
    - branch：main

## 数据集

使用的数据集：[Imagenet](https://image-net.org/)

- 数据集大小：14,197,122张图片
- 数据格式：jpeg
- 下载数据集，并按照[数据集准备说明](../../../onnx/common/prepare_dataset/imagenet/README.md)处理成如下格式。 

```text
data
   ├── n01440764
   │   ├── ILSVRC2012_val_00000293.JPEG
   │   ├── ILSVRC2012_val_00002138.JPEG
   ……
   └── val_map.txt
```

## ViT 推理指南

### 测试方法

在 inference_models 目录下，以 models--google--vit-large-patch16-384 模型为例，进行推理:

```shell
python3 demo/visual/vit/demo_vit.py \
--device gcu \
--model_dir ./model/models--google--vit-large-patch16-384
```

### 参数说明

- device: 模型推理使用的设备，可选 gcu、cpu、cuda
- model_dir: ViT 预训练模型所在的目录
- image_path: image_path 待测试的图片路径，默认为 None，此时程序会自动从网上下载一张图片进行测试

## ViT 性能评估指南

### 测试方法

在 inference_models 目录下，以 models--google--vit-large-patch16-384 模型为例，在 ImageNet 验证集上评估精度和性能:

```shell
python3 demo/visual/vit/benchmark_vit.py \
--model_dir ./model/models--google--vit-large-patch16-384 \
--data_path ./data/val \
--batch_size 256 \
--device gcu \
--warmup_count 20
```

### 参数说明

- device: 模型推理使用的设备，可选 gcu、cpu、cuda
- model_dir: ViT 预训练模型所在的目录
- data_path: ImageNet 验证集所在的路径
- batch_size: batch size 默认为 256，可根据需要调整
- warmup_count: 用于warmup 的 batch 数目

## 模型验证结果示例

- 模型输出参考（性能数据以实测为主）：

### 功能验证结果示例

```
Predicted class: xxxx
```

### 性能测试结果示例

```
{
    "device": xxxx,
    "pretrained model":xxxx,
    "data_path":xxxx,
    "output_dir":xxxx,
    "warmup_count": xxxx,
    "eval_count": xxxx,
    "batch_size": xxxx,
    "acc1":xxxx,
    "FPS": xxxx
}
```
# <span id="bert-for-pytorch">**BERT模型**</span>

本代码仓库参考了[repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)的实现， 在燧原科技第二代训练卡GCU上完成测试。


## <span id="table-of-contents">**目录**</span>
- [BERT](#bert-for-pytorch)
  - [**目录**](#table-of-contents)
  - [**模型介绍**](#model-introduction)
    - [**模型结构**](#model-architecture)
    - [**模型规模**](#default-configuration)
  - [**环境配置**](#environment-setup)
  - [**快速开始**](#start-guide)
    - [**准备训练数据集**](#prepare-dataset)
    - [**准备预训练模型文件**](#prepare-init-checkpoint)
    - [**数据集合**](#collect-all-data)
    - [**开始训练**](#start-fine-tuning-with-the-squad-dataset)
      - [**举例**](#run-bash-examlple)
  - [**结果**](#performance)
    - [**测试命令**](#training-performance-benchmark)
    - [**GCU测试结果**](#gcu-results)
      - [**训练精度**](#training-accuracy-results)
      - [**训练性能**](#training-performance-results)

## <span id="model-introduction">**模型介绍**</span>

本模型基于论文 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 实现.

### <span id="model-architecture">**模型结构**</span>

BERT模型的主体部分由Transformer的解码器(encoder)部分堆叠而成。把词或半词(token)信息、位置信息（token在输入文本中的位置）通过嵌入表(Embedding)映射到隐空间，再输入到堆叠的解码器中，最后接入任务相关的层。Transformer解码器主要包含了多头双向自注意力机制，还包含线性层MLP以及层正则化(layer normalization)等结构。BERT的预训练任务包括完型填空和预测是否下一句两个任务，这两个任务都是自监督任务，无需标注数据。预训练好的模型可以作为微调任务的初始权重，在很多自然语言处理任务中BERT都取得了当时的最佳。

### <span id="default-configuration">**模型规模**</span>
[BERT](https://arxiv.org/abs/1810.04805)一文中提及了两种不同规模的BERT模型，模型的参数如下.
| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|bert-base |12 encoder| 768| 12|4 x  768|512|110M|
|bert-large|24 encoder|1024| 16|4 x 1024|512|330M|

## <span id="environment-setup">**环境配置**</span>

安装配置好GCU驱动和SDK后，本repo模型的其它依赖pip安装即可，如下所示：

    ```bash
    pip install -r requirements.txt
    ```
## <span id="start-guide">**快速开始**</span>

这部分主要内容为如何微调BERT模型，任务为问答任务，数据集为斯坦福数据集squad.

### <span id="prepare-dataset">**准备训练数据集**</span>

squad数据下载地址如下:

-   [SQuAD 1.1](<https://data.deepai.org/squad1.1.zip>)

下载并用命令unzip解压, 解压文件复制到$DATA_PATH/squad目录（move *.json $DATA_PATH/squad）。因为链接中不包含文件 evaluate-v1.1.py, 需要把本repo的文件squad_evaluate-v1.1.py也复制到$DATA_PATH/squad（cp -r ./squad_evaluate-v1.1.py $DATA_PATH/squad）。

### <span id="Prepare init checkpoint">**准备预训练模型文件**</span>

这是模型下载[地址](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/bert/DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1671098413;2535098413&q-key-time=1671098413;2535098413&q-header-list=&q-url-param-list=&q-signature=3a2e2da40d4b9aff631d8fc9efa0cc1c949d712f). 模型文件DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt 下载完成之后放到 $DATA_PATH/checkpoint 目录下。

### <span id="Collect all data">**数据集合**</span>

在上述的数据目录$DATA_PATH中内容如下：

```data
.
├── checkpoint
│   └── DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt
└── squad
    ├── dev-v1.1.json
    ├── squad_evaluate-v1.1.py
    └── train-v1.1.json
```


#### <span id="run-bash-examlple">**举例**</span>
- 在GCU中跑训练squad任务运行脚本:

  ```bash
  bash scripts/run_squad_1gcu.sh
  ```
- 注意事项:
  ```note
  1. 在run_squad_1gcu.sh中DATA_PATH变量替换为自己的路径（如上面所示包含checkpoint和squad）。
  2. export ENFLAME_ENABLE_TF32=true 表示数据类型为TF32, 设置false时数据类型为FP32。
  3. 如果想使用混精增加参数--amp即可，其它更多参数设置参考run_squad.py。
  ```

## <span id="performance">**结果**</span>


## <span id="training-performance-benchmark">**测试命令**</span>

测试脚本如下：

- 对于单卡GCU测试脚本.

  ```bash
  bash scripts/run_squad_1gcu.sh
  ```

- 对于8卡GCU测试脚本.

  ```bash
  bash scripts/run_squad_8gcu.sh
  ```
数据类型为FP32，不开混精。
## <span id="GCU-results">**GCU测试结果**</span>

### <span id="training-accuracy-results">**训练精度**</span>

- 单卡GCU-T20精度测试结果.

| **Epochs** | **Batch Size** | **Accuracy** |
| ---------- | -------------- | ------------------- |
| 2          | 8             | 91.2                |

- 8卡GCU-T20精度测试结果.

| **Epochs** | **Batch Size/GCU** | **Accuracy** |
| ---------- | ------------------ | ------------------- |
| 2          | 8                  | 91.0               |


### <span id="training-performance-results">**训练性能**</span>

- 单卡GCU-T20性能测试结果.

| **Batch Size/GCU** |**Throughput（sentence/s）** |
| -------------- | --------------------- |
| 8             |12.6                     |


- 8卡GCU-T20性能测试结果.

| **Batch Size/GCU** |  **Throughput（sentence/s）** |
| ------------------ |  --------------------- |
| 8                 |  78.1               |

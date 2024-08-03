# Bert
**本文介绍了BERT_base 和 BERT_large 的训练方法。**

## 模型说明
[BERT](https://arxiv.org/abs/1810.04805) （Bidirectional Encoder Representations from Transformers）以[Transformer](https://arxiv.org/abs/1706.03762) 编码器为网络基本组件，使用掩码语言模型（Masked Language Model）和邻接句子预测（Next Sentence Prediction）两个任务在大规模无标注文本语料上进行预训练（pre-train），得到融合了双向内容的通用语义表示模型。以预训练产生的通用语义表示模型为基础，结合任务适配的简单输出层，微调（fine-tune）后即可应用到下游的NLP任务，效果通常也较直接在下游的任务上训练的模型更优。此前BERT即在[GLUE评测任务](https://gluebenchmark.com/tasks)上取得了SOTA的结果。

### 各种大小的BERT配置如下：

|  Model   | num_layers | hidden_size | num_head |paramters|
|  ----  | ----  | --- | ---- | --- |
| BERT_base  | 12 | 768 | 12 | 110M |
| BERT_large  | 24 | 1024 | 16 | 340M |


## 环境准备

* 根据《TopsRider用户使用手册》安装TopsRider软件栈
  * 软件栈安装推荐使用 HOST+Docker 形式。用户下载的 TopsInstaller 安装包中提供了 Dockerfile ，用户可在 Host OS 中完成 Docker image 的编译，详细操作参考《TopsRider用户使用手册》附录部分，完成环境的安装
  * 在使用过程中，已经默认安装了PyTorch、PaddlePaddle、Tensorflow等框架以及相关依赖，用户无需额外安装的安装包

* 完成安装后，进行运行测试。
```
import torch_gcu
torch_gcu.is_available()
```
输入上述命令，在终端输出True，则表示安装成功。
```
True
```
##  数据和模型准备：

### 数据准备

* 下载地址：https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset


* 数据集目录结构：

    ```
    |-- squad
    |   |-- train-v1.1.json
    |   |-- dev-v1.1.json

### 模型准备

* BERT-base预训练权重下载地址：https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_base_pretraining_amp_lamb/files

* BERT-large预训练权重下载地址：https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_large_pretraining_amp_lamb/files

* 下载vocab.txt 和bert_config.json文件： 从https://github.com/google-research/bert 下载bert_base 和bert_large 的checkpoint,从解压出的文件中提取。

## 模型训练

主要参数解释如下

   ```
--device=gcu: 指定使用的设备类型，这里使用的是GCU。
--do_train: 指定执行训练过程。
--do_predict: 指定执行预测过程。
--do_eval: 指定执行评估过程。
--train_batch_size=48: 设置训练时的批量大小为48。
--predict_batch_size=48: 设置预测时的批量大小为48。
--num_train_epochs=2: 设置训练的轮数为2。
--max_steps=-1: 指定训练的最大步数，-1表示根据训练轮数来确定。
--learning_rate=3e-5: 设置学习率为3e-5。
--max_seq_length=384: 设置输入序列的最大长度为384。
--doc_stride=128: 设置文档的滑动窗口步长为128。
--do_lower_case: 指定是否对输入文本进行小写转换（适用于uncased模型）。
--bert_model=bert-base-uncased: 指定使用的BERT模型类型为bert-base-uncased。
--skip_cache: 跳过缓存。
--skip_steps=5: 跳过前5步的训练。
--print_freq=20: 设置打印日志的频率为每20步。
--output_dir=./output: 指定输出目录为./output。
--init_checkpoint=pytorch_bert_base/bert_base_init/bert_base.pt: 指定初始检查点文件。
--train_file=pytorch_bert_base/squad/v1.1/train-v1.1.json: 指定训练数据文件。
--predict_file=pytorch_bert_base/squad/v1.1/dev-v1.1.json: 指定预测数据文件。
--vocab_file=pytorch_bert_base/bert_base_init/vocab.txt: 指定词汇表文件。
--config_file=pytorch_bert_base/bert_base_init/bert_config.json: 指定配置文件。
--eval_script=pytorch_bert_base/squad/v1.1/evaluate-v1.1.py: 指定评估脚本文件。

   ```

### BERT-base 分布式训练
   ```
python3.8 -u -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=34568 \
    --use_env ./run_squad.py \
    --device=gcu \
    --do_train \
    --do_predict \
    --do_eval \
    --train_batch_size=48 \
    --predict_batch_size=48 \
    --num_train_epochs=2 \
    --max_steps=-1 \
    --learning_rate=3e-5 \
    --max_seq_length=384 \
    --doc_stride=128 \
    --do_lower_case \
    --bert_model=bert-base-uncased \
    --skip_cache \
    --skip_steps=5 \
    --print_freq=20 \
    --output_dir=./output \
    --init_checkpoint=pytorch_bert_base/bert_base_init/bert_base.pt \
    --train_file=pytorch_bert_base/squad/v1.1/train-v1.1.json \
    --predict_file=pytorch_bert_base/squad/v1.1/dev-v1.1.json \
    --vocab_file=pytorch_bert_base/bert_base_init/vocab.txt \
    --config_file=pytorch_bert_base/bert_base_init/bert_config.json \
    --eval_script=pytorch_bert_base/squad/v1.1/evaluate-v1.1.py > ${LOG_FILE} 2>&1

   ```


### BERT-large 分布式训练
   ```
   python3.8 -u -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=34568 \
    --use_env ./run_squad.py \
    --device=gcu \
    --do_train \
    --do_eval \
    --do_predict \
    --do_lower_case \
    --train_batch_size=12 \
    --predict_batch_size=12 \
    --num_train_epochs=2 \
    --max_steps=-1 \
    --learning_rate=3e-05 \
    --bert_model=bert-large-uncased \
    --max_seq_length=384 \
    --doc_stride=128 \
    --seed=1 \
    --skip_cache \
    --skip_steps=5 \
    --print_freq=20 \
    --output_dir=./models \
    --init_checkpoint=pytorch_bert_large/checkpoint/DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt \
    --train_file=pytorch_bert_large/squad/v1.1/train-v1.1.json \
    --predict_file=pytorch_bert_large/squad/v1.1/dev-v1.1.json \
    --eval_script=pytorch_bert_large/squad/v1.1/evaluate-v1.1.py \
    --vocab_file=pytorch_bert_large/google_pretrained_weights/vocab.txt \
    --config_file=pytorch_bert_large/google_pretrained_weights/bert_config.json > ${LOG_FILE} 2>&1

   ```

## 模型评估

* 执行收敛验证脚本，模型训练结束后，会自动进行评估

## 训练结果
下面是bert large训练的report：
   ```
{
    ...
    "model": "bert",
    ...
    "predict_num_examples": 10833,
    "predict_batch_size": 12,
    "exact_match": 84.96688741721854,
    "f1": 91.38256891736472
}

   ```
说明： 由于深度学习的训练存在随机性，训练结果可能存在一定差异。

## 引用说明

```
1. Vaswani A, Shazeer N, Parmar N, et al. [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[C]//Advances in Neural Information Processing Systems. 2017: 6000-6010.
2. Devlin J, Chang M W, Lee K, et al. [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805)[J]. arXiv preprint arXiv:1810.04805, 2018.

```

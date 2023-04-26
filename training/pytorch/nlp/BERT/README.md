# BERT For PyTorch

This repository provides comprehensive Bert-large model introduction and script recipe to train model for state-of-the-art performance, tested and maintained by Enflame.


## **Table of Contents**

- [BERT For PyTorch](#bert-for-pytorch)
  - [**Table of Contents**](#table-of-contents)
  - [**Model Introduction**](#model-introduction)
    - [**Model Architecture**](#model-architecture)
    - [Default configuration](#default-configuration)
    - [**Configuration**](#configuration)
  - [**Environment Setup**](#environment-setup)
    - [**Environment Setup On GCU**](#environment-setup-on-gcu)
  - [**Start Guide**](#start-guide)
    - [**Prepare Dataset**](#prepare-dataset)
    - [**Prepare init checkpoint**](#prepare-init-checkpoint)
    - [**Collect all data**](#collect-all-data)
    - [**Start fine-tuning with the SQuAD dataset**](#start-fine-tuning-with-the-squad-dataset)
      - [**Run bash Examlple**](#run-bash-examlple)
  - [**Performance**](#performance)
    - [**Benchmarking**](#benchmarking)
      - [**Training Performance Benchmark**](#training-performance-benchmark)
    - [**GCU Results**](#gcu-results)
      - [**Training Accuracy Results**](#training-accuracy-results)
      - [**Training Performance Results**](#training-performance-results)

## <span id="model-introduction">**Model Introduction**</span>

This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. The full name of Bert is Bidirectional Encoder Representation from Transformers, which is a pre-trained language representation model. It emphasizes that instead of using the traditional one-way language model or shallow splicing of two one-way language models for pre training, it adopts a new masked language model (MLM), so as to generate deep two-way language representation.

### <span id="model-architecture">**Model Architecture**</span>

The BERT model uses the same architecture as the encoder of the Transformer. Input sequences are projected into an embedding space before being fed into the encoder structure. Additionally, positional and segment encodings are added to the embeddings to preserve positional information. The encoder structure is simply a stack of Transformer blocks, which consist of a multi-head attention layer followed by successive stages of feed-forward networks and layer normalization. The multi-head attention layer accomplishes self-attention on multiple input representations.

### Default configuration

The architecture of the BERT model is almost identical to the Transformer model that was first introduced in the [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf). The main innovation of BERT lies in the pre-training step, where the model is trained on two unsupervised prediction tasks using a large text corpus. Training on these unsupervised tasks produces a generic language model, which can then be quickly fine-tuned to achieve state-of-the-art performance on language processing tasks such as question answering.

The BERT paper reports the results for two configurations of BERT, each corresponding to a unique model size. This implementation provides the same configurations by default, which are described in the table below.

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feedforward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERTBASE |12 encoder| 768| 12|4 x  768|512|110M|
|BERTLARGE|24 encoder|1024| 16|4 x 1024|512|330M|

### <span id="configuration">**Configuration**</span>

The default configuration for [Optimizer](#optimizer), [Pre-Processing](#pre-processing) and [Post-Processing](#post-processing) will be highlighted in this section.


## <span id="environment-setup">**Environment Setup**</span>

This section lists the environmental requirements for GCU and GPU enabling the bert model respectively.

### <span id="environment-setup-on-GCU">**Environment Setup On GCU**</span>

1. Basic setup to enable bert model.

   - Some dependencies are alse required listed in requirements.txt.

     ```bash
     pip install -r requirements.txt
     ```


## <span id="start-guide">**Start Guide**</span>

This section will cover more detail how to enabel models in GCU and GPU from scratch respectively, that including but not limited clone source code, prepare dataset, build environment, start training and inference. "dtu" and "_dtu" in the source code refer to gcu.


### <span id="prepare-dataset">**Prepare Dataset**</span>

This repository provides scripts to download, verify, and extract the following datasets:

-   [SQuAD 1.1](<https://data.deepai.org/squad1.1.zip>)

Download and unzip, move *.json  to $DATA_PATH/squad. For the link do not include evaluate-v1.1.py, we should copy and  squad_evaluate-v1.1.py in this repo to $DATA_PATH/squad

### **Prepare init checkpoint**
If you want to use a pre-trained checkpoint, visit [here](https://topsmodel-1257133546.cos.ap-shanghai.myqcloud.com/topsmodel-1257133546/topsegc/local/model/bert/DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt?q-sign-algorithm=sha1&q-ak=AKIDYyBAwXzDD1e4GEzZUBgy2iDU5TeaIVUG&q-sign-time=1671098413;2535098413&q-key-time=1671098413;2535098413&q-header-list=&q-url-param-list=&q-signature=3a2e2da40d4b9aff631d8fc9efa0cc1c949d712f). This downloaded checkpoint is used to fine-tune on SQuAD. Ensure you unzip the downloaded file and place the checkpoint in the `checkpoints/` folder. Save DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt to $DATA_PATH/checkpoint.


### **Collect all data**
Collect all data to dataset path($DATA_PATH) as above, dataset directory contain filles like:

```data
.
├── checkpoint
│   └── DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt
└── squad
    ├── dev-v1.1.json
    ├── squad_evaluate-v1.1.py
    └── train-v1.1.json
```


### <span id="start-training">**Start fine-tuning with the SQuAD dataset**</span>

#### **Run bash Examlple**

- This fine-tuning for question answering with the SQuAD task, run command:

  ```bash
  bash scripts/run_squad_1gcu.sh
  ```
- Note:
  ```note
  1. change DATA_PATH to your own path include checkpoint and squad as above.
  2. export ENFLAME_ENABLE_TF32=true means use TF32 dtype, and use FP32 dtype if set false.
  3. if want to use mix precision(fp16 and TF32 or FP32), add para --amp in *.sh
  ```

## <span id="performance">**Performance**</span>

### <span id="benchmarking">**Benchmarking**</span>

#### <span id="training-performance-benchmark">**Training Performance Benchmark**</span>

To benchmark the training performance benchmark on a spcific batch size, run command:

- For 1 [GCU] and [TF32].

  ```bash
  bash scripts/run_squad_1gcu.sh
  ```

- For multiple [GCU] and [TF32].

  ```bash
  bash scripts/run_squad_8gcu.sh
  ```

### <span id="GCU-results">**GCU Results**</span>

#### <span id="training-accuracy-results">**Training Accuracy Results**</span>

- Training on 1 GCU T20 under different dtype with fixed batchsize.

  Our results were obtrained by running the [Training Performance Benchmark](#training-performance-benchmark) on Enflame 1xGCU T20.

| **Epochs** | **Batch Size** | **Accuracy - TF32** |
| ---------- | -------------- | ------------------- |
| 2          | 8             | 91.2                |

- Training accuracy(top1) multiple GCU T20 under different dtype with fixed batchsize.

  Our results were obtrained by running the [Training Performance Benchmark](#training-performance-benchmark) on Enflame 8xGCU T20.

| **Epochs** | **Batch Size/GCU** | **Accuracy - TF32** |
| ---------- | ------------------ | ------------------- |
| 2          | 8                  | 91.0               |



#### <span id="training-performance-results">**Training Performance Results**</span>

- Training performance(Throughput) for 1 GCU T20 under different dtype with various batch size.

  Our results were obtrained by running the [Training Performance Benchmark](#training-performance-benchmark) on Enfalme 1xGCU T20. Performance(images per second) were average over an entire training epoch.

| **Batch Size/GCU** |**Throughput - TF32** |
| -------------- | --------------------- |
| 8             |12.6                     |


- Training performance(Throughput) for multiple GCU T20 under different dtype with various batch size.

  Our results were obtrained by running the [Training Performance Benchmark](#training-performance-benchmark) on Enflame 8xGCU T20. Performance(images per second) were average over an entire training epoch.

| **Batch Size/GCU** |  **Throughput - TF32** |
| ------------------ |  --------------------- |
| 8                 |  78.1               |

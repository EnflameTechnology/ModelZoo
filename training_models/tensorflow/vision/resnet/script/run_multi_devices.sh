#!/usr/bin/env bash
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1 --tf_xla_min_cluster_size=4"
export ENFLAME_UMD_FLAGS="ib_pool_size=134217728"
export ENFLAME_MEMUSE_POLICY=performance

export ENFLAME_DEVICE_MODE=ONEDEVICE_EX
export ENFLAME_COMPILE_OPTIONS_HLIR="hlir-pipeline{tensor-split=true}"
export ENFLAME_CLUSTER_PARALLEL=true
export ENFLAME_ENABLE_TF32=true

mpirun -np <chip numbers> --allow-run-as-root python run_classify.py \
    --is_training=True \
    --device=<[dtu|gpu]> \
    --depth=50 \
    --dtype=<[fp32|bf16]> \
    --data_format=NCHW \
    --data_dir=./dataset \
    --dataset=imagenet \
    --batch_size=<batch_size> \
    --epoch=90 \
    --resnet_version=1.5 \
    --enable_saver=True \
    --enable_evaluate=True \
    --optimizer=momentum \
    --enable_horovod=True
#!/usr/bin/env bash

set -eu -o pipefail
# Copyright 2021-2022 Enflame. All Rights Reserved.
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export TF_XLA_FLAGS="--tf_xla_auto_jit=-1 --tf_xla_min_cluster_size=4"
export ENFLAME_UMD_FLAGS="ib_pool_size=134217728"
export ENFLAME_MEMUSE_POLICY=performance

# export XLA_FLAGS=" --xla_dump_hlo_as_text --xla_dump_to=hlo_dump --xla_dump_hlo_pass_re='.*'"
export ENFLAME_DEVICE_MODE=ONEDEVICE_EX
export ENFLAME_COMPILE_OPTIONS_HLIR="hlir-training-pipeline{disable-passes=HlirCustomFusionPass tensor-split=true}"
export ENFLAME_CLUSTER_PARALLEL=true
export ENFLAME_ENABLE_TF32=true

RESULTS_DIR='/results'
CHECKPOINTS_DIR='/results/checkpoints'
STAT_FILE=${RESULTS_DIR}/DGX1_fp32_static_dtu.json
mkdir -p $CHECKPOINTS_DIR

PREC=${1:-'fp32'}
SEED=${2:-1}
LR=${3:-0.0006}
WARMUP=${4:-4000}
NUM_EPOCHS=${5:-40}
BATCH_SIZE=${6:-8192}
NUM_GPU=${7:-1}
: ${USE_SLURM:=0}

DISTRIBUTED="-m torch.distributed.launch --nproc_per_node=${NUM_GPU}"
[ ${USE_SLURM} = 1 ] && DISTRIBUTED+=" --nnodes ${WORLD_SIZE} --node_rank ${SLURM_NODEID}  \
        --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} "

if [ "$PREC" = "amp" ];
then
    PREC='--amp --amp-level O2 '
else
    PREC=''
fi

LOG_TIME=`date '+%Y_%m_%d_%H_%M_%S'`
LOG_NAME=transformer_logs/log_${LOG_TIME}.log
python -u transformer/train.py \
  ./transformer_data/wmt14_en_de_joined_dict \
  --arch transformer \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-6 \
  --warmup-updates ${WARMUP} \
  --lr $LR \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens ${BATCH_SIZE} \
  --max-sentences-valid 72\
  --max-sentences 72\
  --max-source-positions 128\
  --max-target-positions 128\
  --skip-invalid-size-inputs-valid-test\
  --seed ${SEED} \
  --max-epoch ${NUM_EPOCHS} \
  --no-epoch-checkpoints \
  --device_type gcu\
  --log-interval 1\
  --static-batch True\
  --training_step_per_epoch 10\
  --eval_step_per_epoch 10\
  --skip-steps 2\
  --save-dir ${RESULTS_DIR} \
  --stat-file ${STAT_FILE} >> ${LOG_NAME} 2>&1


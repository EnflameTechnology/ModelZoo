#!/bin/bash
#
# Copyright 2022 Enflame. All Rights Reserved.
#
RESULTS_DIR='/results'
CHECKPOINTS_DIR='/results/checkpoints'
STAT_FILE=${RESULTS_DIR}/DGX1_fp32_static_gcu.json
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
LOG_NAME=transformer_logs/log_cpu_${LOG_TIME}.log
python -u transformer/train.py \
  ./transformer_data/wmt14_en_de_joined_dict \
  --arch transformer \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates ${WARMUP} \
  --lr $LR \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens ${BATCH_SIZE} \
  --max-sentences-valid 96\
  --max-sentences 96\
  --max-source-positions 128\
  --max-target-positions 128\
  --skip-invalid-size-inputs-valid-test\
  --seed ${SEED} \
  --max-epoch ${NUM_EPOCHS} \
  --no-epoch-checkpoints \
  --device_type cpu \
  --log-interval 1 \
  --static-batch True\
  --training_step_per_epoch 4\
  --eval_step_per_epoch 4\
  --skip-steps 2\
  --save-dir ${RESULTS_DIR} \
  --stat-file ${STAT_FILE} >> ${LOG_NAME} 2>&1

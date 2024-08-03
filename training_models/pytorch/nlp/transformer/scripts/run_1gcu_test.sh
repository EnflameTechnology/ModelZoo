#! /bin/bash
export ENFLAME_ENABLE_TF32=true

DATA_PATH=./transformer_data_pytorch
python3 -u train.py \
    $DATA_PATH/wmt14_en_de_joined_dict \
    --share-all-embeddings \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints \
    --device=gcu  \
    --device_type=gcu \
    --max-tokens=18432 \
    --max-sentences-valid=144 \
    --max-sentences=144 \
    --max-epoch=1 \
    --training_step_per_epoch=200 \
    --eval_step_per_epoch=5 \
    --warmup-init-lr=1e-09 \
    --warmup-updates=4000 \
    --lr=0.0006 \
    --max-source-positions=128 \
    --max-target-positions=128 \
    --seed=1 \
    --log-interval=10 \
    --static-batch=True \
    --log-get-freq=10 \
    --skip-steps=5 \
    --save-interval-updates=20000 \
    --model=transformer \
    --arch=transformer_wmt_en_de_base_t2t \
    --stat-file=DGX1_fp32_static_gcu.json \
    --save-dir=runs

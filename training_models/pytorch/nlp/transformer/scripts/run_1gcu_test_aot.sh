#! /bin/bash
export ENFLAME_ENABLE_TF32=true
export ENFLAME_ENABLE_PT_JIT_AOT_MIXED="true"
export COMPILE_OPTIONS_MLIR_DBG=" "
export ENFLAME_PT_AOT_CFG="use_sync=false use_jit_aot=true use_scalar_cache=true use_view_cascade=true use_op_check= use_pt= use_aot= use_jit="


DATA_PATH=./transformer_data_pytorch
python3 -u train.py \
    $DATA_PATH/wmt14_en_de_joined_dict \
    --share-all-embeddings \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints \
    --device=gcu  \
    --device_type=gcu \
    --max-tokens=5120 \
    --max-sentences-valid=1024 \
    --max-sentences=1024 \
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
    --use_aot=True \
    --log-get-freq=10 \
    --skip-steps=5 \
    --save-interval-updates=20000 \
    --model=transformer \
    --arch=transformer_wmt_en_de_base_t2t \
    --stat-file=DGX1_fp32_static_gcu.json \
    --save-dir=runs

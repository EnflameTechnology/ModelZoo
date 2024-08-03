#! /bin/bash
export ENFLAME_ENABLE_TF32=true

DATA_PATH=./transformer_data_pytorch
python3 -u -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env train.py \
    ./transformer_data_pytorch/wmt14_en_de_joined_dict \
    --model=transfomer \
    --device=gcu \
    --warmup-init-lr=1e-09 \
    --warmup-updates=4000 \
    --lr=0.0006 \
    --dropout=0.0 \
    --max-tokens=18432 \
    --max-sentences-valid=144 \
    --max-sentences=144 \
    --max-source-positions=128 \
    --max-target-positions=128 \
    --seed=1 \
    --max-epoch=1 \
    --device_type=gcu \
    --log-interval=200 \
    --static-batch=True \
    --training_step_per_epoch=200 \
    --eval_step_per_epoch=5 \
    --log-get-freq=20 \
    --skip-steps=5 \
    --save-dir=runs \
    --stat-file=DGX1_fp32_static_gcu.json \
    --share-all-embeddings \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints


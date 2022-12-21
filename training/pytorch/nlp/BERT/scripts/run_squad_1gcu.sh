#!/usr/bin/env bash
export ENFLAME_ENABLE_TF32=true

DATA_PATH=dataset/pytorch_bert_large
# note: if use mix precision, add para --amp
python -u run_squad.py \
    --device=gcu \
    --train_batch_size=8 \
    --predict_batch_size=8 \
    --num_train_epochs=2 \
    --max_steps=-1 \
    --learning_rate=3.0e-05 \
    --skip_steps=5 \
    --print_freq=20 \
    --do_train \
    --do_eval \
    --do_predict \
    --do_lower_case \
    --bert_model=bert-large-uncased \
    --max_seq_length=384 \
    --doc_stride=128 \
    --seed=1 \
    --init_checkpoint=$DATA_PATH/checkpoint/DLE_BERT_FP16_PyT_LAMB_92_hard_scaling_node.pt \
    --train_file=$DATA_PATH/squad/train-v1.1.json \
    --predict_file=$DATA_PATH/squad/dev-v1.1.json \
    --eval_script=$DATA_PATH/squad/squad_evaluate-v1.1.py \
    --vocab_file=vocab/vocab \
    --config_file=bert_config.json \
    --output_dir=models


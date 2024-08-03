#!/bin/bash
set -eu -o pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ENV_PATH="$(dirname "$(dirname "$(dirname "$(dirname "${SCRIPT_DIR}")")")")"/Models/envs
LOG_PATH=${SCRIPT_DIR}/logs/
DATASET_DIR="$(dirname "$(dirname "$(dirname "${SCRIPT_DIR}")")")"/dataset
MODEL_DIR="$(dirname "$(dirname "$(dirname "$(dirname "${SCRIPT_DIR}")")")")"/Models/General_Model/official
MODEL_EXECUTION_DIR=${MODEL_DIR}/pytorch_lazy/nlp/BERT

source ${ENV_PATH}/Enflame_pt_efp_performance_distribute_ddp.sh



env

if [ ! -d "${LOG_PATH}" ] ; then
    mkdir -p "${LOG_PATH}"
fi
LOG_TIME=$(date '+%Y%m%d%H%M%S')

LOG_FILE=${LOG_PATH}/enflame_distributed_8card_pytorch_bert_base_convergence_efp_${LOG_TIME}.log

cd ${MODEL_EXECUTION_DIR} || (echo "No valid model execution dir: ${MODEL_EXECUTION_DIR}" && exit 1)
# Check whether the dataset exists in DATASET_DIR
function check_dataset() {
local target=$1
local tree=$2
if [ ! -e $target ];then
    cat << EOF
ERROR: $target does not exist.
Please make sure the following files or directories are in the ${DATASET_DIR}:$tree
EOF
    exit 1
fi
}
check_dataset ${DATASET_DIR}/pytorch_bert_base '
    pytorch_bert_base
    ├── bert_base_init
    └── squad
'
# Create soft link for dataset
function create_symlink() {
local src=$1
local dst=$2
if [ $# -ge 3 ]; then
    local dst_dir=$3
    if [ "$dst_dir" = 'isfile' ]; then
        if [ $# -eq 4 ];then
            local file_dst_dir=$4
            if [ ! -d $file_dst_dir ]; then
                mkdir -p $file_dst_dir
            fi
            ln -sf $src $dst
        else
            if [ ! -f $dst ]; then
                ln -sf $src $dst
            fi
        fi
    else
        if [ ! -d $dst ]; then
            if [ ! -d $dst_dir ]; then
                mkdir -p $dst_dir
            fi
            ln -sf $src $dst
        fi
    fi
else
    if [ ! -d $dst ];then
        ln -sf $src $dst
    fi
fi
}
create_symlink ${DATASET_DIR}/pytorch_bert_base pytorch_bert_base
python3.8 -m pip install -r requirements.txt

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

exit $?

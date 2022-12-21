#!/bin/bash
onnx_path=${1:-"path_onnx_model"}
path_of_pretrained_bert_model=${2:-"path_of_pretrained_bert_model"}
path_of_squad_data=${3:-"path_of_squad_data"}
onnx_version=${4:-"nvidia"}
run_device=${5:-"cpu"}
batch_size=${6:-"1"}
seq_length=${7:-"384"}
doc_stride=${8:-"128"}
squad_version=${9:-"1.1"}

# set the bert base pretrained files for inference
export BERT_DIR=$path_of_pretrained_bert_model

# set squad 1.1 data preprocess files for inference
export SQUAD_DIR=$path_of_squad_data
if [ "$squad_version" = "1.1" ] ; then
    version_2_with_negative="False"
else
    version_2_with_negative="True"
fi


RESULTS_DIR=.

echo "Squad directory set as " $SQUAD_DIR " BERT directory set as " $BERT_DIR
echo "Results directory set as " $RESULTS_DIR


DATESTAMP=`date +'%y%m%d%H%M%S'`

for DIR_or_file in $SQUAD_DIR $RESULTS_DIR $BERT_DIR/vocab.txt $BERT_DIR/bert_config.json; do
  if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
     echo "Error! $DIR_or_file directory missing. Please mount correctly"
     exit 0
  fi
done

python3 run_onnx.py \
--vocab_file=$BERT_DIR/vocab.txt \
--bert_config_file=$BERT_DIR/bert_config.json \
--onnx_path=$onnx_path \
--model_ver=$onnx_version \
--device=$run_device \
--do_predict=True \
--predict_file=$SQUAD_DIR/dev-v${squad_version}.json \
--eval_script=$SQUAD_DIR/evaluate-v${squad_version}.py \
--predict_batch_size=$batch_size \
--max_seq_length=$seq_length \
--doc_stride=$doc_stride \
--output_dir=$RESULTS_DIR \
--logging_file=${DATESTAMP}.log \
--version_2_with_negative=${version_2_with_negative}
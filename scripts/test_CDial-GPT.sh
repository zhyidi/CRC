#!/bin/bash
stage=$3
LOGDIR=run_logs/test_stage${stage}
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

use_tcp=$2

if [ "$use_tcp" = true ]; then
    LOGPATH="test_CDial-GPT_useTCP.log"
else
    LOGPATH="test_CDial-GPT.log"
fi

# tcp_path="None"
tcp_path="outputs/TCP/best_model_test.txt"

if [ "$stage" = 1 ]; then
    test_path="data/sample_test.json"
    max_length=80
elif [ "$stage" = 2 ]; then
    test_path="data/sample_test.json"
    max_length=120
else
    test_path="data/sample_test_CDial-GPT_stage3.json"
    max_length=80
fi

cache_dir="caches/CDial-GPT"
model_dir="logs/CDial-GPT"
output_dir="outputs/CDial-GPT"
test_batch_size=4

device=$1

python backbones/CDial-GPT/run_infer.py --use_tcp ${use_tcp} \
        --tcp_path ${tcp_path} \
        --test_path ${test_path} \
        --cache_dir ${cache_dir} \
        --model_dir ${model_dir} \
        --output_dir ${output_dir} \
        --test_batch_size ${test_batch_size} \
        --max_dec_len ${max_length} \
        --device ${device} \
        --stage ${stage} \
        > ${LOGDIR}/${LOGPATH} 2>&1 
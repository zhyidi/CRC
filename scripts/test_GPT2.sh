#!/bin/bash
stage=$3
LOGDIR=run_logs/test_stage${stage}
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

use_tcp=$2

if [ "$use_tcp" = true ]; then
    LOGPATH="test_GPT2_useTCP.log"
else
    LOGPATH="test_GPT2.log"
fi

# tcp_path="None"
tcp_path="outputs/TCP/best_model_test.txt"

if [ "$stage" = 1 ]; then
    test_path="data/sample_test.json"
    max_dec_len=80
elif [ "$stage" = 2 ]; then
    test_path="data/sample_test.json"
    max_dec_len=120
else
    test_path="data/sample_test_GPT2_stage3.json"
    max_dec_len=80
fi

cache_dir="caches/GPT2"
model_dir="logs/GPT2"
output_dir="outputs/GPT2"
test_batch_size=4

device=$1

python backbones/GPT-2/run_infer.py --use_tcp ${use_tcp} \
        --tcp_path ${tcp_path} \
        --test_path ${test_path} \
        --cache_dir ${cache_dir} \
        --model_dir ${model_dir} \
        --output_dir ${output_dir} \
        --test_batch_size ${test_batch_size} \
        --max_dec_len ${max_dec_len} \
        --device ${device} \
        --stage ${stage} \
        > ${LOGDIR}/${LOGPATH} 2>&1 
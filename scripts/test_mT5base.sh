#!/bin/bash
stage=$3
LOGDIR=run_logs/test_stage${stage}
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

use_tcp=$2

if [ "$use_tcp" = true ]; then
    LOGPATH="test_mT5base_useTCP.log"
else
    LOGPATH="test_mT5base.log"
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
    test_path="data/sample_test_mT5base_stage3.json"
    max_dec_len=80
fi

cache_dir="caches/mT5base"
model_dir="logs/mT5base"
output_dir="outputs/mT5base"
test_batch_size=4

device=$1

python backbones/mT5base/run_infer.py --use_tcp ${use_tcp} \
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
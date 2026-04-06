#!/bin/bash
LOGDIR=run_logs/test
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

# dataset
test_data="data/sample_test_Rec.json"
bert_dir="config/bert-base-chinese"
cache_dir="caches/TCP"
log_dir="logs/TCP"

# decode args
output_dir="outputs/TCP"
use_ssd="false"
test_batch_size=6

# output_dir="outputs/TCP_SSD"
# use_ssd="true"
# test_batch_size=1

python main.py --mode test \
    --test_data ${test_data} \
    --bert_dir ${bert_dir} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --output_dir ${output_dir} \
    --use_ssd ${use_ssd} \
    --test_batch_size ${test_batch_size} \
    > ${LOGDIR}/test_TCP.log 2>&1
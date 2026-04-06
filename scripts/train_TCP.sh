#!/bin/bash
LOGDIR=run_logs/train
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi
LOGPATH="train_TCP.log"

# dataset
train_data="data/sample_train_Rec.json"
dev_data="data/sample_dev.json"
bert_dir="config/bert-base-chinese"
cache_dir="caches/TCP"
log_dir="logs/TCP"

# train args
num_epochs=10
batch_size=6

python main.py --mode train \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --bert_dir ${bert_dir} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    > ${LOGDIR}/${LOGPATH} 2>&1 

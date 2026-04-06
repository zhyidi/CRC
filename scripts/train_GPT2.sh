#!/bin/bash
stage=$4
LOGDIR=run_logs/train_stage${stage}
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

use_tcp=$3

if [ "$use_tcp" = true ]; then
    LOGPATH="train_GPT2_useTCP.log"
else
    LOGPATH="train_GPT2.log"
fi

if [ "$stage" = 1 ]; then
    train_path="data/sample_train.json"
    valid_path="data/sample_dev.json"
    model_checkpoint="config/gpt2-chinese-cluecorpussmall"
    n_epochs=10
    gradient_accumulation_steps=8
elif [ "$stage" = 2 ]; then
    train_path="data/sample_train_with_RQ_from_GPT2_w_TCP_pred_train.json"
    valid_path="data/sample_train_with_RQ_from_GPT2_w_TCP_pred_dev.json"
    model_checkpoint="logs/GPT2_w_TCP_stage1"
    n_epochs=10
    gradient_accumulation_steps=8
else
    train_path="data/sample_train_QOptimization_from_GPT2_w_TCP_pred_train.json"
    valid_path="data/sample_train_QOptimization_from_GPT2_w_TCP_pred_dev.json"
    model_checkpoint="logs/GPT2_w_TCP_stage1"
    n_epochs=10
    gradient_accumulation_steps=8
fi

cache_dir="caches/GPT2"
log_dir="logs/GPT2"
lr=5e-5
warmup_steps=3000
train_batch_size=8
valid_batch_size=8

# python backbones/GPT-2/run_train.py --use_tcp ${use_tcp} \
CUDA_VISIBLE_DEVICES=$2 python -m torch.distributed.launch --nproc_per_node=$1 --nnodes=1 backbones/GPT-2/run_train.py --use_tcp ${use_tcp} \
        --stage ${stage} \
        --train_path ${train_path} \
        --valid_path ${valid_path} \
        --cache_dir ${cache_dir} \
        --model_checkpoint ${model_checkpoint} \
        --log_dir ${log_dir} \
        --n_epochs ${n_epochs} \
        --lr ${lr} \
        --warmup_steps ${warmup_steps} \
        --train_batch_size ${train_batch_size} \
        --valid_batch_size ${valid_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        > ${LOGDIR}/${LOGPATH} 2>&1 
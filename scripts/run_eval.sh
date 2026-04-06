#!/bin/bash
LOGDIR=eval_result/stage3
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

python eval/eval_dialogue.py --eval_file outputs/GPT2_w_TCP_stage3/output_test.txt --gold_file data/sample_test.json > ${LOGDIR}/eval_dialogue_GPT_TCP_stage3.log 2>&1 &
python eval/eval_dialogue.py --eval_file outputs/CDial-GPT_w_TCP_stage3/output_test.txt --gold_file data/sample_test.json > ${LOGDIR}/eval_dialogue_CDial-GPT_TCP_stage3.log 2>&1 &
python eval/eval_dialogue.py --eval_file outputs/BART_w_TCP_stage3/output_test.txt --gold_file data/sample_test.json > ${LOGDIR}/eval_dialogue_BART_TCP_stage3.log 2>&1 &
python eval/eval_dialogue.py --eval_file outputs/mT5base_w_TCP_stage3/output_test.txt --gold_file data/sample_test.json > ${LOGDIR}/eval_dialogue_mT5base_TCP_stage3.log 2>&1 &


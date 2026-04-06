#!/bin/bash
#SBATCH --job-name=Task_P4D
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:TeslaV100-SXM2-32GB:1
#SBATCH --cpus-per-task=2
#SBATCH --output=stage123_e1010_m1.log

run_train_sh(){
  bash scripts/train_GPT2.sh ${GPUNUM} ${GPUID} ${use_tcp} ${stage}
  bash scripts/train_mT5base.sh ${GPUNUM} ${GPUID} ${use_tcp} ${stage}
  bash scripts/train_CDial-GPT.sh ${GPUNUM} ${GPUID} ${use_tcp} ${stage}
  bash scripts/train_BART.sh ${GPUNUM} ${GPUID} ${use_tcp} ${stage}
}

GPUNUM=2
GPUID=0,1
use_tcp=true
stage=1
run_train_sh
stage=2
run_train_sh
stage=3
run_train_sh

run_test_sh(){
  # bash scripts/test_GPT2.sh ${GPUID} ${use_tcp} $((${stage}-1))
  # bash scripts/test_CDial-GPT.sh ${GPUID} ${use_tcp} $((${stage}-1))
  bash scripts/test_GPT2.sh ${GPUID} ${use_tcp} ${stage}
  bash scripts/test_CDial-GPT.sh ${GPUID} ${use_tcp} ${stage}
  python data/utils/get_test_data_stage3.py outputs/GPT2_w_TCP_stage2/output_test.txt GPT2
  python data/utils/get_test_data_stage3.py outputs/CDial-GPT_w_TCP_stage2/output_test.txt CDial-GPT
  bash scripts/test_GPT2.sh ${GPUID} ${use_tcp} $((${stage}+1))
  bash scripts/test_CDial-GPT.sh ${GPUID} ${use_tcp} $((${stage}+1))

  # bash scripts/test_BART.sh ${GPUID} ${use_tcp} $((${stage}-1))
  # bash scripts/test_mT5base.sh ${GPUID} ${use_tcp} $((${stage}-1))
  bash scripts/test_BART.sh ${GPUID} ${use_tcp} ${stage}
  bash scripts/test_mT5base.sh ${GPUID} ${use_tcp} ${stage}
  python data/utils/get_test_data_stage3.py outputs/BART_w_TCP_stage2/output_test.txt BART
  python data/utils/get_test_data_stage3.py outputs/mT5base_w_TCP_stage2/output_test.txt mT5base
  bash scripts/test_BART.sh ${GPUID} ${use_tcp} $((${stage}+1))
  bash scripts/test_mT5base.sh ${GPUID} ${use_tcp} $((${stage}+1))
}

GPUNUM=1
GPUID=cuda:0
use_tcp=true
stage=2
run_test_sh


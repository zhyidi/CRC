#!/bin/bash
#SBATCH --job-name=Task_one
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:2
#SBATCH --cpus-per-task=2
#SBATCH --output=llama3_lora.log


llamafactory-cli train examples/train_lora/llama3_lora_sft_Plan4RecDial_1.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft_stage1.yaml

llamafactory-cli train examples/train_lora/llama3_lora_sft_Plan4RecDial_2.yaml
llamafactory-cli train examples/train_lora/llama3_lora_predict_Plan4RecDial_2.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft_stage2.yaml

llamafactory-cli train examples/train_lora/llama3_lora_sft_Plan4RecDial_3.yaml
python get_test_data_stage3_llama.py
llamafactory-cli train examples/train_lora/llama3_lora_predict_Plan4RecDial_3.yaml
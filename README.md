# Training and Inference

For fine-tuning and inference on GPT2, CDial-GPT, BART, mT5:

```bash
cd Code_main_directory
bash scripts/run.sh
```

For fine-tuning and inference on LLaMA3-8B: 

1. Download [LLaMA-Factory_address](https://github.com/hiyouga/LLaMA-Factory)

2. After adding the configuration file under backbone\llama-factory-Configuration:

   ```bash
   cd Code_main_directory
   bash run.sh
   ```

# Evaluate

```bash
bash scripts/run_eval.sh
```


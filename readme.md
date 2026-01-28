# Overview

This repository provides reproducible code and configurations for all experiments reported in our paper ("Demystifying Design Choices of Reinforcement Fine-tuning: A Batched Contextual Bandit Learning Perspective"). We implement multiple training settings under a unified framework, with example scripts provided for each experimental setup.



## Environment Setting

Our implementation is based on the **VeRL** framework. Please follow the official installation guide to set up the environment:

https://verl.readthedocs.io/en/latest/start/install.html

Our requirement.txt previous the running packages.



## Hardware and Software

All experiments are conducted on a single NVIDIA A100 GPU with 40GB memory.

The software environment is based on Ubuntu~22.04,

with Python~3.12, PyTorch~2.5.1, and CUDA~12.4.

The host machine is equipped with 10 vCPUs

(Intel Xeon Processor, Skylake with IBRS support).



## Dataset

- **GSM8K**: https://huggingface.co/datasets/openai/gsm8k

- **MATH**: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

  

## Base Model

* **Qwen2.5-0.5B-Instruct**: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
* **LLaMA-3.2-1B-Instruct**: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
* **OLMo-2-0425-1B-Instruct**: https://huggingface.co/unsloth/OLMo-2-0425-1B-Instruct



## Key Code Structure

```
Bandit-Reinforcement-Fine-tuning/
├── Dataset/
│   ├── gsm8k/
│   └── MATH/
├── Running-Bash/  
├── Running-Results/
    ├── Figure/
    ├── result/
├── Verl/
│   └── trainer/
│       ├── main_ppo_grpo.py
│       ├── main_ppo_bandit.py
│       ├── main_ppo_replay.py
│       └── ppo/
│           ├── ray_trainer_grpo.py
│           ├── ray_trainer_bandit.py
│           ├── ray_trainer_replay.py
│           ├── core_algos_grpo.py
│           ├── core_algos_bandit.py
│           └── core_algos_replay.py
├── requirements.txt
  
```



## Running Bash

We have six model-dataset pairs:

Qwen-GSM, LLaMA-GSM, OLMo-GSM, Qwen-MATH, LLaMA-MATH, and OLMo-MATH.

Take Qwen-GSM as Example:

bash run_grpo_0032_8_qwen0.5b_gsm.sh 

All bash of  Qwen-GSM can see at Runing-Bash 



## Running Results

All experiment can see at Running-Results.


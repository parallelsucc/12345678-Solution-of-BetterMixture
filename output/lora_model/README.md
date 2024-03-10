---
license: other
base_model: baichuan-inc/Baichuan2-7B-Base
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: lora_model
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lora_model

This model is a fine-tuned version of [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) on the /root/autodl-tmp/dj_mixture_challenge/output/sft_data/mixture.jsonl dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 64
- total_train_batch_size: 256
- total_eval_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.31.0
- Pytorch 2.1.0+cu118
- Datasets 2.18.0
- Tokenizers 0.13.3

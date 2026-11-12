# Full fine-tuning vs LoRA vs QLoRA: GPT-2 small on Tiny Shakespeare (cuda)

| Metric | Full fine-tune | LoRA | QLoRA (4-bit) |
|---|---:|---:|---:|
| Trainable parameters | 124,439,808 (100.0%) | 1,179,648 (0.94%) | 1,179,648 (0.94%) |
| Frozen projection weights | trained | fp32 | NF4 4-bit |
| GPU memory after model load (MiB) | 475 | 480 | 217 |
| Peak GPU memory allocated (MiB) | 2,818 | 1,865 | 1,603 |
| Peak GPU memory reserved (MiB) | 3,184 | 2,112 | 1,640 |
| Peak process RSS (MiB) | 1,638 | 1,638 | 1,638 |
| Time/step, median [p10–p90] (ms) | 325 [322–326] | 218 [218–219] | 296 [292–303] |
| Validation loss, step 0 | 4.3267 | 4.3267 | 4.4802 |
| Validation loss, final | 3.3799 | 3.4131 | 3.4378 |
| Validation perplexity, final | 29.37 | 30.36 | 31.12 |
| Peak learning rate | 3e-05 | 0.001 | 0.001 |

Hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4096 MiB), driver 610.47; AMD Ryzen 5 5600H with Radeon Graphics, 6.7 GiB RAM, WSL2; torch 2.14.1+cu126 (CUDA 12.6), peft 0.21.2, Python 3.12.3.

Setup: 300 steps, batch 4 x 256 tokens, AdamW (weight decay 0.01), 20-step linear warmup then constant LR, grad clip 1.0, seed 1337. LoRA rank 8, alpha 16, dropout 0 on attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj. QLoRA: the same LoRA on top of those projections stored in NF4 (block 64, fp32 absmax, no double quantization), dequantized on the fly, fp32 compute; embeddings, LayerNorms and biases stay fp32. Validation = all 132 non-overlapping 256-token windows of the held-out split. Time/step = synchronized forward+backward+clip+optimizer step, first 10 steps and evaluation excluded. GPU peak memory is measured after model load, over training and periodic evaluation; process RSS is the whole process, including model load.

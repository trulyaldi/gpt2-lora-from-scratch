# LoRA vs full fine-tuning: GPT-2 small on Tiny Shakespeare (cpu)

| Metric | Full fine-tune | LoRA |
|---|---:|---:|
| Trainable parameters | 124,439,808 (100.0%) | 1,179,648 (0.94%) |
| Peak GPU memory allocated (MiB) | n/a | n/a |
| Peak GPU memory reserved (MiB) | n/a | n/a |
| Peak process RSS (MiB) | 4,138 | 3,339 |
| Time/step, median [p10–p90] (ms) | 4808 [4735–4860] | 3659 [3626–3687] |
| Validation loss, step 0 | 4.3267 | 4.3267 |
| Validation loss, final | 3.6461 | 3.6833 |
| Validation perplexity, final | 38.32 | 39.78 |
| Peak learning rate | 3e-05 | 0.001 |

Hardware: CPU only, 6 torch threads; AMD Ryzen 5 5600H with Radeon Graphics, 6.7 GiB RAM, WSL2; torch 2.15.0+cu132 (CUDA 13.2), peft 0.21.2, Python 3.12.3.

Setup: 20 steps, batch 4 x 256 tokens, AdamW (weight decay 0.01), 20-step linear warmup then constant LR, grad clip 1.0, seed 1337. LoRA rank 8, alpha 16, dropout 0 on attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj. Validation = all 132 non-overlapping 256-token windows of the held-out split. Time/step = synchronized forward+backward+clip+optimizer step, first 5 steps and evaluation excluded. GPU peak memory is measured after model load, over training and periodic evaluation; process RSS is the whole process, including model load.

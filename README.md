# GPT-2 from scratch: NumPy, PyTorch and LoRA

I wrote GPT-2 small (124M parameters) in plain NumPy, with every backward pass derived by hand, to understand exactly what a transformer computes during training. I then ported it to PyTorch, checked the port against the NumPy version, and wrote LoRA from scratch and checked it against Hugging Face PEFT. Finally, I measured what LoRA actually saves compared with full fine-tuning when adapting pretrained GPT-2 to Tiny Shakespeare.

## What is verified, and how

All checks are pytest tests in `tests/` (19 tests).

**Handwritten gradients** (`test_gradcheck.py`). Each NumPy layer is compared with central finite differences in float64. That covers Linear, GELU, MLP, softmax, cross-entropy, LayerNorm, attention, embedding and the transformer block, plus every parameter of a small 2-layer GPT, including the tied output head. Every gradient entry is checked, and the relative error must be below 1e-7; the worst measured is 4.8e-9. One full-width GPT-2 small block (768 wide, 12 heads) is spot-checked on sampled entries with a 1e-6 threshold; the worst measured is 8.3e-8. The same file checks that attention is causal and that the initial loss of GPT-2 small is within 0.25 of the expected ln(50257) + 768·0.02²/2 ≈ 10.98. It also checks that the NumPy AdamW can overfit a single batch.

**NumPy vs PyTorch** (`test_numpy_torch_parity.py`). Both models get identical float64 weights. Logits, the attention output and the loss must match to 1e-11. The gradient of every parameter must match to rtol 1e-8, atol 1e-10.

**LoRA vs PEFT** (`test_lora_peft_parity.py`). My `LoRALinear` and PEFT's `LoraModel` are given the same base weights and the same A and B matrices, in float64. Trainable parameter counts must be equal. Outputs and loss must match to 1e-11. LoRA gradients, the outputs after one SGD step, and the merged weights must match to 1e-10. `test_lora.py` also checks that inserting adapters leaves outputs exactly unchanged, that only A and B train, and that save/load and merging work.

**Pretrained weights** (`test_pretrained.py`). The PyTorch model loaded from OpenAI's GPT-2 checkpoint must produce the same logits as Hugging Face `GPT2LMHeadModel`, within rtol 1e-4 and atol 5e-4 in float32, and the same argmax at every position. This test is skipped until the weights are in the local Hugging Face cache. Running the benchmark downloads them.

## Results

Both runs start from pretrained GPT-2 small and fine-tune on Tiny Shakespeare. The data is 304,222 training tokens and 33,803 validation tokens, tokenized with GPT-2 BPE and split contiguously 90/10.

Training settings are shared by both runs:
- batch 4 × 256 tokens, 300 AdamW steps, gradient clipping at 1.0, and the same seeded batches;
- learning rate 3e-5 for full fine-tuning and 1e-3 for LoRA;
- LoRA rank 8, alpha 16, on the attention and MLP projections of all 12 blocks.

Each run is a separate process. Validation loss is measured on all 132 non-overlapping 256-token windows of the held-out split.

Hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB, driver 610.47), AMD Ryzen 5 5600H, 6.7 GiB RAM visible to WSL2, PyTorch 2.15.0 (CUDA 13.2), PEFT 0.21.2.

| GPU, 300 steps | Full fine-tune | LoRA (r=8) |
|---|---:|---:|
| Trainable parameters | 124,439,808 (100%) | 1,179,648 (0.94%) |
| Peak GPU memory allocated | 2,746 MiB | 1,800 MiB |
| Time per step, median | 328 ms | 223 ms |
| Validation loss, before training | 4.3267 | 4.3267 |
| Validation loss, after 300 steps | 3.3799 | 3.4131 |

Time per step covers forward, backward, clipping and the optimizer step, with the GPU synchronized. Evaluation and the first 10 steps are excluded. Peak GPU memory is measured after the model is loaded and includes the resident weights.

**CPU vs GPU.** The same benchmark was run on the CPU for 20 steps with the same batch, to measure time per step only:

| Time per step, median | Full fine-tune | LoRA |
|---|---:|---:|
| GPU (RTX 3050 Ti) | 328 ms | 223 ms |
| CPU (Ryzen 5 5600H, 6 torch threads) | 4,808 ms | 3,659 ms |
| CPU peak process memory (RSS) | 4,138 MiB | 3,339 MiB |

The NumPy model fine-tunes the same pretrained weights on the CPU too. Its run used batch 2 × 128 for 20 steps, so the numbers aren't comparable with the tables above. It took 3.1 s per step with 2,805 MiB peak memory. Validation loss on 16 held-out windows went from 4.196 to 3.597.

Raw results, per-step losses, plots and the full hardware record are in `results/shakespeare_gpt2/`, `results/shakespeare_gpt2_cpu/` and `results/finetune_numpy/`.

## How to run

```bash
pip install -r requirements.txt     # the Linux torch wheel includes CUDA (several GB)
pytest                              # 19 passed (18 passed, 1 skipped before GPT-2 weights are cached)
python benchmark.py                 # LoRA vs full fine-tune on GPU, ~4 min
python benchmark.py --device cpu --steps 20 --eval-every 20 --timing-skip 5   # CPU timing run
python finetune_numpy.py            # fine-tune pretrained GPT-2 in pure NumPy on CPU, 20 steps
```

The scripts download Tiny Shakespeare (1.1 MB) into `data/` and the GPT-2 weights (550 MB) into the Hugging Face cache on first use.

On WSL2, if torch segfaults on CUDA initialization, a native Linux NVIDIA driver installed inside WSL is probably conflicting with the Windows driver. A workaround is to run `export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(dirname "$(ls -t /usr/lib/wsl/drivers/*/libcuda.so.1.1 | head -1)")"` first.

## Files

| File | Contents |
|---|---|
| `gpt_numpy.py` | GPT-2 in NumPy: layers with handwritten backward passes, AdamW, sampling, safetensors loader |
| `gpt_torch.py` | PyTorch port of the same model, NumPy-to-PyTorch weight copy, pretrained loader |
| `lora.py` | LoRA layer, apply/merge, adapter save/load |
| `common.py` | Tiny Shakespeare loader and machine description shared by the scripts |
| `benchmark.py` | LoRA vs full fine-tuning; writes JSON, a Markdown table and plots to `results/` |
| `finetune_numpy.py` | Fine-tunes pretrained GPT-2 with the NumPy model; writes `results/finetune_numpy/` |
| `tests/` | Gradient checks, NumPy/PyTorch parity, LoRA/PEFT parity, LoRA behaviour, pretrained loading |
| `results/` | Saved benchmark and fine-tuning results |
| `requirements.txt` | Pinned dependencies |
| `pytest.ini` | Lets `pytest` import the modules from the repository root |
| `LICENSE` | MIT license |

## Limitations

These are single runs with one seed. The 0.033 validation-loss gap between full fine-tuning and LoRA is a single short run, not a measured confidence interval, and could change with longer training, tuned learning rates or a different rank. The learning rates are common defaults, not tuned.

All timings come from one laptop GPU and CPU under WSL2. The batch was chosen so that full fine-tuning fits in 4 GB of VRAM. Larger batches would likely widen the memory gap, but full fine-tuning doesn't fit on this GPU at larger batches, so that wasn't measured.

The NumPy model is for correctness and understanding, not speed. It has no KV cache, and the PyTorch port uses an explicit attention mask instead of a fused attention kernel.

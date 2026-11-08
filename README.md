# GPT-2 from scratch: NumPy, PyTorch and LoRA

GPT-2 small (124M parameters) written in plain NumPy with every backward pass derived by hand, then ported to PyTorch, extended with LoRA written from scratch, and used to measure what LoRA saves compared with full fine-tuning. Every implementation is tested against an independent reference.

## Key results

- **Correct gradients.** Every handwritten gradient matches finite differences; worst relative error 4.8e-9.
- **Matching implementations.** The PyTorch port matches the NumPy model to 1e-11, my LoRA matches Hugging Face PEFT to 1e-10, and the pretrained model matches Hugging Face's GPT-2.
- **LoRA vs full fine-tuning on a 4 GB laptop GPU.** LoRA trained 0.95% of GPT-2's parameters, used 34% less GPU memory and 32% less time per step, and reached a validation loss of 3.41, against 3.38 for full fine-tuning.

## Results

Pretrained GPT-2 small fine-tuned on Tiny Shakespeare (304,222 training and 33,803 validation tokens, contiguous 90/10 split).

| RTX 3050 Ti Laptop (4 GB), 300 steps | Full fine-tune | LoRA (rank 8) |
|---|---:|---:|
| Trainable parameters | 124,439,808 | 1,179,648 (0.95%) |
| Peak GPU memory | 2,746 MiB | 1,800 MiB |
| Time per step (median) | 328 ms | 223 ms |
| Validation loss, before → after | 4.327 → 3.380 | 4.327 → 3.413 |

The memory saving is almost exactly the AdamW state for 124M parameters (about 950 MiB). The frozen weights and the activations still have to fit, which is why the saving is 34% and not more at this batch size.

On the CPU (Ryzen 5 5600H, 6 threads, 20 steps), a step takes 4,808 ms for full fine-tuning and 3,659 ms for LoRA, about 15 times slower than the GPU. The NumPy model also fine-tunes the pretrained weights on the CPU, at 3.1 s per step with a smaller batch (2 × 128).

**Setup.** Both runs use the same seeded batches of 4 × 256 tokens, AdamW (weight decay 0.01) with a 20-step warmup, gradient clipping at 1.0, and learning rates of 3e-5 (full) and 1e-3 (LoRA). LoRA uses rank 8 and alpha 16 on the attention and MLP projections of all 12 blocks. Each run is a separate process. Validation covers all 132 non-overlapping 256-token windows of the held-out split. Timings are synchronized, exclude evaluation and the first 10 steps, and were taken with PyTorch's deterministic mode on.

Raw data, per-step losses, plots and the full hardware record are in `results/`.

## How it is verified

19 pytest tests in `tests/`:

| Test | What it checks | Tolerance |
|---|---|---|
| `test_gradcheck.py` | Every NumPy layer and a full 2-layer GPT against finite differences (float64); causal masking; expected initial loss; overfitting a single batch | relative error < 1e-7 |
| `test_numpy_torch_parity.py` | Same weights in both models: logits, loss and every parameter gradient | 1e-11 (outputs), 1e-8 (gradients) |
| `test_lora_peft_parity.py` | My LoRA against PEFT with identical weights: outputs, gradients, one optimizer step, merged weights | 1e-10 |
| `test_lora.py` | Adapters leave outputs unchanged at start, only A and B train, save/load and merging work | exact |
| `test_pretrained.py` | Loaded OpenAI weights give the same logits and predictions as Hugging Face `GPT2LMHeadModel` | 1e-4 (float32) |

## Quick start

```bash
pip install -r requirements.txt   # Linux torch wheel includes CUDA (several GB)
pytest                            # 19 tests; the pretrained test is skipped until weights are cached
python benchmark.py               # LoRA vs full fine-tuning on the GPU, about 4 min
python benchmark.py --device cpu --steps 20 --eval-every 20 --timing-skip 5   # CPU timing
python finetune_numpy.py          # fine-tune pretrained GPT-2 in pure NumPy, 20 steps
```

Tiny Shakespeare (1.1 MB) and the GPT-2 weights (550 MB) download automatically on first use.

## Files

| File | Contents |
|---|---|
| `gpt_numpy.py` | GPT-2 in NumPy with handwritten backward passes, AdamW and a weight loader |
| `gpt_torch.py` | PyTorch port and pretrained-weight loader |
| `lora.py` | LoRA layer, apply, merge, save and load |
| `benchmark.py` | LoRA vs full fine-tuning; writes JSON, a table and plots to `results/` |
| `finetune_numpy.py` | Fine-tuning with the NumPy model |
| `common.py` | Dataset loader and hardware description |
| `tests/` | The tests above |

## Limitations

- Single runs with one seed and untuned learning rates. The 0.033 gap in validation loss could change with longer training, other learning rates or another rank.
- All timings come from one laptop under WSL2. The batch is limited by fitting full fine-tuning into 4 GB, so larger batches, where LoRA's memory advantage would grow, were not measured.
- The NumPy model is built for correctness, not speed. The PyTorch port uses an explicit attention mask, not a fused attention kernel.

**WSL2 note.** If torch crashes when CUDA starts, a Linux NVIDIA driver installed inside WSL is probably conflicting with the Windows driver. Run this first:
`export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(dirname "$(ls -t /usr/lib/wsl/drivers/*/libcuda.so.1.1 | head -1)")"`

## License

MIT

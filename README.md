# GPT-2 from scratch: NumPy, PyTorch, LoRA and QLoRA

I wrote GPT-2 small (124M parameters) in plain NumPy, with every backward pass derived by hand, to understand exactly what a transformer computes during training. I then ported it to PyTorch, checked the port against the NumPy version, and wrote LoRA from scratch and checked it against Hugging Face PEFT. I added QLoRA, LoRA on top of 4-bit NF4 base weights, and checked its quantization against bitsandbytes. Finally, I measured what LoRA and QLoRA actually save compared with full fine-tuning when adapting pretrained GPT-2 to Tiny Shakespeare on a GPU.

## What is verified, and how

All checks are pytest tests in `tests/` (24 tests).

**Handwritten gradients** (`test_gradcheck.py`). Each NumPy layer is compared with central finite differences in float64. That covers Linear, GELU, MLP, softmax, cross-entropy, LayerNorm, attention, embedding and the transformer block, plus every parameter of a small 2-layer GPT, including the tied output head. Every gradient entry is checked, and the relative error must be below 1e-7; the worst measured is 4.8e-9. One full-width GPT-2 small block (768 wide, 12 heads) is spot-checked on sampled entries with a 1e-6 threshold; the worst measured is 8.3e-8. The same file checks that attention is causal and that the initial loss of GPT-2 small is within 0.25 of the expected ln(50257) + 768·0.02²/2 ≈ 10.98. It also checks that the NumPy AdamW can overfit a single batch.

**NumPy vs PyTorch** (`test_numpy_torch_parity.py`). Both models get identical float64 weights. Logits, the attention output and the loss must match to 1e-11. The gradient of every parameter must match to rtol 1e-8, atol 1e-10.

**LoRA vs PEFT** (`test_lora_peft_parity.py`). My `LoRALinear` and PEFT's `LoraModel` are given the same base weights and the same A and B matrices, in float64. Trainable parameter counts must be equal. Outputs and loss must match to 1e-11. LoRA gradients, the outputs after one SGD step, and the merged weights must match to 1e-10. `test_lora.py` also checks that inserting adapters leaves outputs exactly unchanged, that only A and B train, and that save/load and merging work.

**QLoRA vs bitsandbytes** (`test_qlora.py`). My NF4 code table must equal bitsandbytes' exactly. On GPT-2-shaped weights, my per-block scales must equal bitsandbytes' exactly. My 4-bit codes must also match its codes, except for values that fall within float32 rounding of the midpoint between two levels, which its CUDA kernel can round either way. That is about 1 value in 15 million, and the test allows at most 1 in a million. The 4-bit LoRA layer must give the same outputs and gradients as plain LoRA on the dequantized weights, in float64. The hand-written backward pass is checked with `torch.autograd.gradcheck`. Another check confirms that backward keeps only the packed 4-bit codes and scales, not a full-precision copy of the weights.

**Pretrained weights** (`test_pretrained.py`). The PyTorch model loaded from OpenAI's GPT-2 checkpoint must produce the same logits as Hugging Face `GPT2LMHeadModel`, within rtol 1e-4 and atol 5e-4 in float32, and the same argmax at every position. This test is skipped until the weights are in the local Hugging Face cache. Running the benchmark downloads them.

## Results

All three runs start from pretrained GPT-2 small and fine-tune on Tiny Shakespeare. The data is 304,222 training tokens and 33,803 validation tokens, tokenized with GPT-2 BPE and split contiguously 90/10.

Training settings are shared by all runs:
- batch 4 × 256 tokens, 300 AdamW steps, gradient clipping at 1.0, and the same seeded batches;
- learning rate 3e-5 for full fine-tuning and 1e-3 for LoRA and QLoRA;
- LoRA rank 8, alpha 16, on the attention and MLP projections of all 12 blocks;
- for QLoRA, those projections (85M of the 124M weights) are stored in 4-bit NF4 with one fp32 scale per 64 values and dequantized on the fly. Embeddings, LayerNorms and biases stay in fp32. All runs compute in fp32, so QLoRA differs from LoRA only in how the frozen weights are stored.

Each run is a separate process. Validation loss is measured on all 132 non-overlapping 256-token windows of the held-out split.

Hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB, driver 610.47), AMD Ryzen 5 5600H, 6.7 GiB RAM visible to WSL2, PyTorch 2.14.1 (CUDA 12.6), PEFT 0.21.2.

| GPU, 300 steps | Full fine-tune | LoRA (r=8) | QLoRA (r=8, NF4) |
|---|---:|---:|---:|
| Trainable parameters | 124,439,808 (100%) | 1,179,648 (0.94%) | 1,179,648 (0.94%) |
| GPU memory for the model | 475 MiB | 480 MiB | 217 MiB |
| Peak GPU memory allocated | 2,818 MiB | 1,865 MiB | 1,603 MiB |
| Time per step, median | 325 ms | 218 ms | 296 ms |
| Validation loss, before training | 4.3267 | 4.3267 | 4.4802 |
| Validation loss, after 300 steps | 3.3799 | 3.4131 | 3.4378 |

Time per step covers forward, backward, clipping and the optimizer step, with the GPU synchronized. Evaluation and the first 10 steps are excluded. Peak GPU memory is measured after the model is loaded and includes the resident weights.

QLoRA cuts the model's own GPU memory from 480 to 217 MiB. Peak memory drops by 262 MiB (14%) compared with LoRA, because at this batch size activations take most of the memory, not weights. QLoRA starts from a higher loss: quantizing the weights to 4 bits alone raises validation loss from 4.3267 to 4.4802, and bitsandbytes' NF4 gives the same 4.4802. Training recovers most of that, and QLoRA ends 0.025 above LoRA. It is 36% slower per step than LoRA because my layers dequantize the weights with ordinary PyTorch operations in every forward and backward pass. bitsandbytes uses fused CUDA kernels for this, so its speed would differ. Double quantization of the scales and paged optimizers from the QLoRA paper are not implemented; at this model size they would save only a few MiB.

The NumPy model fine-tunes the same pretrained weights on the CPU. Its run used batch 2 × 128 for 20 steps, so the numbers aren't comparable with the table above. It took 3.1 s per step with 2,805 MiB peak memory. Validation loss on 16 held-out windows went from 4.196 to 3.597.

Raw results, per-step losses, plots and the full hardware record are in `results/shakespeare_gpt2/` and `results/finetune_numpy/`.

## How to run

```bash
pip install -r requirements.txt     # the Linux torch wheel includes CUDA (several GB)
pytest                              # 24 passed (the pretrained test is skipped until GPT-2 weights are cached,
                                    #   the bitsandbytes comparison without CUDA)
python benchmark.py                 # full fine-tune vs LoRA vs QLoRA on GPU, ~7 min
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
| `qlora.py` | NF4 quantization, 4-bit frozen linear layer, QLoRA layer |
| `common.py` | Tiny Shakespeare loader and machine description shared by the scripts |
| `benchmark.py` | Full fine-tuning vs LoRA vs QLoRA on GPU; writes JSON, a Markdown table and plots to `results/` |
| `finetune_numpy.py` | Fine-tunes pretrained GPT-2 with the NumPy model; writes `results/finetune_numpy/` |
| `tests/` | Gradient checks, NumPy/PyTorch parity, LoRA/PEFT parity, LoRA behaviour, QLoRA/bitsandbytes checks, pretrained loading |
| `results/` | Saved benchmark and fine-tuning results |
| `requirements.txt` | Pinned dependencies |
| `pytest.ini` | Lets `pytest` import the modules from the repository root |
| `LICENSE` | MIT license |

## Limitations

These are single runs with one seed. The validation-loss gaps between the methods (0.033 from full fine-tuning to LoRA, 0.025 from LoRA to QLoRA) come from single short runs, not measured confidence intervals, and could change with longer training, tuned learning rates or a different rank. The learning rates are common defaults, not tuned.

All benchmark timings come from one laptop GPU under WSL2. The batch was chosen so that full fine-tuning fits in 4 GB of VRAM. Larger batches would likely widen the memory gap, but full fine-tuning doesn't fit on this GPU at larger batches, so that wasn't measured.

The NumPy model is for correctness and understanding, not speed. It has no KV cache, and the PyTorch port uses an explicit attention mask instead of a fused attention kernel.

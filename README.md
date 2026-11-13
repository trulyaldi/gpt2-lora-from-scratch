# GPT-2 from scratch: NumPy, PyTorch, LoRA and QLoRA

GPT-2 small (124M parameters) written in plain NumPy with every backward pass derived by hand, then ported to PyTorch, extended with LoRA and 4-bit QLoRA written from scratch, and used to measure what LoRA and QLoRA save compared with full fine-tuning on a GPU. Every implementation is tested against an independent reference.

## Key results

- **Correct gradients.** Every handwritten gradient matches finite differences; worst relative error 4.8e-9.
- **Matching implementations.** The PyTorch port matches the NumPy model to 1e-11, my LoRA matches Hugging Face PEFT to 1e-10, my 4-bit NF4 quantization matches bitsandbytes, and the pretrained model matches Hugging Face's GPT-2.
- **LoRA vs full fine-tuning on a 4 GB laptop GPU.** LoRA trained 0.95% of GPT-2's parameters, used 34% less GPU memory and 33% less time per step, and reached a validation loss of 3.41, against 3.38 for full fine-tuning.
- **QLoRA vs LoRA.** Storing the frozen weights in 4 bits cut the model's GPU memory from 480 to 217 MiB and peak memory by a further 14%. QLoRA reached a validation loss of 3.44 and was 36% slower per step than LoRA.

## Results

**Legacy run.** The numbers in this section and in *Key results* come from the original 300-step, single-seed, untuned benchmark on a 90/10 split, with no separate test set (`results/shakespeare_gpt2/`). The current benchmark is `benchmark.py`. It uses an 80/10/10 split, learning-rate search on validation loss, 3 seeds, and test-set evaluation of the best-validation checkpoint. See [Multi-seed benchmark](#multi-seed-benchmark) and `BENCHMARK_REPORT.md`.

Pretrained GPT-2 small fine-tuned on Tiny Shakespeare (304,222 training and 33,803 validation tokens, contiguous 90/10 split).

| RTX 3050 Ti Laptop (4 GB), 300 steps | Full fine-tune | LoRA (rank 8) | QLoRA (rank 8, NF4) |
|---|---:|---:|---:|
| Trainable parameters | 124,439,808 | 1,179,648 (0.95%) | 1,179,648 (0.95%) |
| GPU memory for the model | 475 MiB | 480 MiB | 217 MiB |
| Peak GPU memory | 2,818 MiB | 1,865 MiB | 1,603 MiB |
| Time per step (median) | 325 ms | 218 ms | 296 ms |
| Validation loss, before → after | 4.327 → 3.380 | 4.327 → 3.413 | 4.480 → 3.438 |

The memory saving is almost exactly the AdamW state for 124M parameters (about 950 MiB). The frozen weights and the activations still have to fit, which is why the saving is 34% and not more at this batch size.

QLoRA stores the 85M attention and MLP projection weights in 4-bit NF4, with one fp32 scale per 64 values, and dequantizes them on the fly. Embeddings, LayerNorms and biases stay in fp32. Its peak memory is 262 MiB below LoRA's; most of the remaining memory is activations, not weights. Quantization alone raises the starting validation loss from 4.327 to 4.480 (bitsandbytes' NF4 gives the same 4.480), and training recovers most of it. QLoRA is slower than LoRA because my layers dequantize with ordinary PyTorch operations in every forward and backward pass, where bitsandbytes uses fused CUDA kernels. Double quantization and paged optimizers from the QLoRA paper are not implemented; at this size they would save only a few MiB.

Separately, the NumPy model fine-tunes the pretrained weights on the CPU, at 3.1 s per step with a smaller batch (2 × 128).

**Setup.** All runs use the same seeded batches of 4 × 256 tokens, AdamW (weight decay 0.01) with a 20-step warmup, gradient clipping at 1.0, and learning rates of 3e-5 (full) and 1e-3 (LoRA and QLoRA). All compute in fp32, so QLoRA differs from LoRA only in how the frozen weights are stored. LoRA and QLoRA use rank 8 and alpha 16 on the attention and MLP projections of all 12 blocks. Each run is a separate process. Validation covers all 132 non-overlapping 256-token windows of the held-out split. Timings are synchronized, exclude evaluation and the first 10 steps, and were taken with PyTorch's deterministic mode on. PyTorch 2.14.1 (CUDA 12.6).

Raw data, per-step losses, plots and the full hardware record are in `results/shakespeare_gpt2/`.

## Multi-seed benchmark

`benchmark.py` runs the comparison in four stages, each resumable:

1. **tune**: one run per learning rate per method on tuning seed 0. Full fine-tuning tries 1e-5, 3e-5 and 1e-4; LoRA and QLoRA each try 3e-4, 1e-3 and 3e-3. The LR with the lowest best-checkpoint validation loss is selected.
2. **final**: seeds 42, 123 and 456, each with the selected LR, keeping the checkpoint with the lowest validation loss.
3. **test**: reload each checkpoint, check that its validation loss matches the recorded value, then measure test loss.
4. **report**: mean ± std, CSV/JSON, figures in `results/figures/` and `BENCHMARK_REPORT.md`.

Data handling:
- The raw text is cut into contiguous 80/10/10 sections at blank lines, and each section is tokenized separately. Training windows never cross a split.
- Training and tuning never load the test split.

Every method uses the same settings:
- the same pretrained weights, batches, effective batch (8 × 256 tokens, as 2 micro-batches of 4), and number of steps (1000 by default)
- warmup followed by cosine decay
- the same evaluation points and loss

Each run is its own process.

Failure handling:
- A failed run, including CUDA out-of-memory, is recorded in `failed.json`, listed in the report, and makes the command exit non-zero.
- A run whose reserved memory exceeds the GPU's physical memory is treated as out-of-memory. On WSL2 the driver otherwise spills silently into system RAM. In one forced test (full fine-tuning, batch 16 in a single micro-batch), the run reserved 7.4 GB on the 4 GB GPU and took 11.3 s per step instead of failing.

```bash
python benchmark.py all --preset smoke   # ~9 min: 2 LRs x 2 seeds x 20 steps, results/smoke/ (pipeline check only)
python benchmark.py all                  # default: 3 LRs + 3 seeds x 1000 steps per method, several hours
python benchmark.py all --steps 2000 --seeds 1 2 3 4 5   # any option can be overridden
```

## How it is verified

41 pytest tests in `tests/`:

| Test | What it checks | Tolerance |
|---|---|---|
| `test_gradcheck.py` | Every NumPy layer and a full 2-layer GPT against finite differences (float64); causal masking; expected initial loss; overfitting a single batch | relative error < 1e-7 |
| `test_numpy_torch_parity.py` | Same weights in both models: logits, loss and every parameter gradient | 1e-11 (outputs), 1e-8 (gradients) |
| `test_lora_peft_parity.py` | My LoRA against PEFT with identical weights: outputs, gradients, one optimizer step, merged weights | 1e-10 |
| `test_lora.py` | Adapters leave outputs unchanged at start, only A and B train, save/load and merging work | exact |
| `test_qlora.py` | NF4 code table and quantization against bitsandbytes; QLoRA against LoRA on the dequantized weights; the custom backward against `gradcheck`; backward keeps no full-precision weights | exact codes and scales, except values within float32 rounding of a level midpoint (bitsandbytes' kernel rounds those either way, about 1 in 15 million); 1e-10 (gradients) |
| `test_benchmark.py` | 80/10/10 split is contiguous and tokenized per split; batch order depends only on the seed; full fine-tuning updates pretrained weights; LoRA/QLoRA freeze the base (bitwise) and update only A and B, with optimizer state for adapters only; gradient accumulation equals one large batch; evaluation and statistics; on CUDA, a short run of each method loads no test data and its checkpoint reloads | 1e-10 / exact |
| `test_pretrained.py` | Loaded OpenAI weights give the same logits and predictions as Hugging Face `GPT2LMHeadModel` | 1e-4 (float32) |

## Quick start

```bash
pip install -r requirements.txt   # Linux torch wheel includes CUDA (several GB)
pytest                            # 41 tests; the pretrained test is skipped until weights are cached,
                                  #   the bitsandbytes comparison without CUDA
python benchmark.py all --preset smoke   # benchmark pipeline check, about 9 min (see above for the full run)
python finetune_numpy.py          # fine-tune pretrained GPT-2 in pure NumPy, 20 steps
```

Tiny Shakespeare (1.1 MB) and the GPT-2 weights (550 MB) download automatically on first use.

## Files

| File | Contents |
|---|---|
| `gpt_numpy.py` | GPT-2 in NumPy with handwritten backward passes, AdamW and a weight loader |
| `gpt_torch.py` | PyTorch port and pretrained-weight loader |
| `lora.py` | LoRA layer, apply, merge, save and load |
| `qlora.py` | NF4 quantization, 4-bit frozen linear layer and QLoRA layer |
| `benchmark.py` | Benchmark CLI: LR tuning, multi-seed final runs, test evaluation, report |
| `training.py` | The single training loop used by all three methods, evaluation, checkpointing and resume |
| `analysis.py` | Mean ± std across seeds, CSV/JSON, figures and `BENCHMARK_REPORT.md` |
| `finetune_numpy.py` | Fine-tuning with the NumPy model |
| `common.py` | Dataset splits, seeding and hardware description |
| `tests/` | The tests above |

## Limitations

- The legacy results above are single runs with one seed and untuned learning rates. The validation-loss gaps (0.033 from full fine-tuning to LoRA, 0.025 from LoRA to QLoRA) could change with longer training, other learning rates or another rank.
- All timings come from one laptop GPU under WSL2. The batch is limited by fitting full fine-tuning into 4 GB, so larger batches, where LoRA's memory advantage would grow, were not measured.
- The NumPy model is built for correctness, not speed. The PyTorch port uses an explicit attention mask, not a fused attention kernel.

**WSL2 note.** If torch crashes when CUDA starts, a Linux NVIDIA driver installed inside WSL is probably conflicting with the Windows driver. Run this first:
`export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(dirname "$(ls -t /usr/lib/wsl/drivers/*/libcuda.so.1.1 | head -1)")"`

## License

MIT

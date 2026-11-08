# Execution status (2026-11-08)

Environment: RTX 3050 Ti Laptop GPU (4 GB), WSL2, Python 3.12.3, torch 2.15.0+cu132, peft 0.21.2, numpy 2.5.4.
Dependencies were installed into `.venv` from `requirements-lora.txt`.

## Tests: 5 passed, 0 skipped

```
tests/test_lora.py::test_lora_freeze_initial_grad_merge_save PASSED
tests/test_lora.py::test_lora_shapes PASSED
tests/test_peft.py::test_peft_forward_grad_update_merge PASSED
tests/test_pretrained.py::test_pretrained_gpt2_matches_huggingface PASSED
tests/test_torch_port.py::test_numpy_port_forward_attention_loss_gradients PASSED
```

- **NumPy ↔ PyTorch:** logits, attention output and loss agree to 1e-11, and every parameter gradient (biases included) agrees to 1e-8 relative, all in float64.
- **Custom LoRA ↔ PEFT:** trainable-parameter counts are equal; outputs, loss, gradients, one SGD step and merged weights agree to 1e-10 in float64.
- **Pretrained loader ↔ HuggingFace `GPT2LMHeadModel`:** max logit difference is 7.6e-5 in float32, and argmax agrees on every position.
- **Mutation checks:** I broke the GELU approximation, perturbed the LoRA scaling by 1e-6 relative, and offset the copied biases by 1e-6. Each change made the suite fail.

Fixes made to get there:

- `test_torch_port.py` had a forward hook that returned `dict.setdefault(...)`, which replaced the attention output with a NumPy array. The hook now returns `None`.
- Bias-gradient checks were added to the same test.

## Benchmark: results/shakespeare_gpt2/

Run with `python benchmark.py` (about 4 minutes). See `comparison.md` there, or the README table.

## Known environment issue

A native Linux NVIDIA 580 driver is installed inside WSL at `/lib/x86_64-linux-gnu/libcuda.so.580.178.04`. Its `libnvidia-ptxjitcompiler` is loaded by the Windows WSL driver, and the version mismatch makes torch segfault on CUDA initialization, including CPU-only `backward()`. All commands above were run with this workaround:

    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(dirname "$(ls -t /usr/lib/wsl/drivers/*/libcuda.so.1.1 | head -1)")"

The native libraries come from the `libnvidia-compute-580` package. The permanent fix is `sudo apt remove libnvidia-compute-580`, but check apt's prompt, because dependents such as `nvidia-cuda-toolkit` may be removed too.

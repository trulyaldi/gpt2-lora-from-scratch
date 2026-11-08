# GPT-2 from Scratch in NumPy

A from-scratch implementation of **GPT-2 Small (124M parameters)** using NumPy, with manually implemented forward and backward passes—without PyTorch, TensorFlow, or automatic differentiation.

The goal is to understand transformer architectures, gradient computation, and language-model training from first principles.

## Implementation

The project implements the core components of GPT-2:

- Token and positional embeddings
- Causal multi-head self-attention
- Layer normalization and residual connections
- Feed-forward networks with GELU activation
- Cross-entropy loss and manual backpropagation
- Autoregressive text generation

The training pipeline includes **AdamW**, gradient clipping, gradient accumulation, learning-rate warmup, and cosine decay.

## Validation

Analytical gradients are checked against numerical gradients using central finite differences:

$$
\frac{\partial L}{\partial \theta_i}
\approx
\frac{L(\theta_i+\varepsilon)-L(\theta_i-\varepsilon)}
{2\varepsilon}.
$$

Additional sanity checks include:

- Verifying the initial cross-entropy loss.
- Overfitting a single training batch.
- Loading pretrained GPT-2 weights into the NumPy implementation.

## Training

The model supports fine-tuning pretrained GPT-2 Small on **Tiny Shakespeare**.

The project also examines the computational cost of transformer training, including FLOPs per token and memory requirements.

## Purpose

This project focuses on **mathematical correctness, transparent implementations, and understanding transformer training internals**, rather than optimizing for production-scale performance.

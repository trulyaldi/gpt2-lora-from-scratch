"""Fine-tune pretrained GPT-2 small on Tiny Shakespeare using only the NumPy model (CPU).

    python finetune_numpy.py                  # 20 steps, writes results/finetune_numpy/results.json

Uses gpt_numpy's handwritten forward/backward, AdamW, gradient clipping and warmup + cosine
schedule. Validation loss is measured on the same fixed held-out windows before and after.
"""
import argparse
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np

import gpt_numpy as g
from common import ROOT, gpt2_weights_path, load_shakespeare, system_info


def parse_args():
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--sequence-length', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--val-windows', type=int, default=16, help='held-out windows used for validation loss')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--output', type=Path, default=ROOT / 'results' / 'finetune_numpy')
    return p.parse_args()


def validation_loss(model, val, length, n_windows):
    """Mean cross-entropy over the first n non-overlapping windows of the validation split."""
    ce = g.CrossEntropyLoss()
    losses = []
    for start in range(0, n_windows * length, length):
        x, y = val[None, start:start + length], val[None, start + 1:start + length + 1]
        losses.append(ce.forward(model.forward(x), y))
    return float(np.mean(losses))


def main():
    """Load pretrained GPT-2 into the NumPy model, fine-tune, evaluate, sample, and save results."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    train, val, _ = load_shakespeare()

    model = g.GPT(dtype=np.float32)
    g.load_hf_gpt2(model, gpt2_weights_path())
    params = model.parameters()
    optimizer = g.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    ce = g.CrossEntropyLoss()

    initial = validation_loss(model, val, args.sequence_length, args.val_windows)
    print(f'step 0: validation loss {initial:.4f}')
    step_times, train_losses = [], []
    for step in range(args.steps):
        starts = rng.integers(0, len(train) - args.sequence_length - 1, size=args.batch_size)
        x = np.stack([train[i:i + args.sequence_length] for i in starts])
        y = np.stack([train[i + 1:i + args.sequence_length + 1] for i in starts])
        t0 = time.perf_counter()
        optimizer.lr = g.get_lr(step, max_lr=args.lr, min_lr=args.lr / 10,
                                warmup=max(1, args.steps // 10), max_steps=args.steps)
        model.zero_grad()
        loss = ce.forward(model.forward(x), y)
        model.backward(ce.backward())
        g.clip_grad_norm(params, 1.0)
        optimizer.step()
        step_times.append(time.perf_counter() - t0)
        train_losses.append(float(loss))
        print(f'step {step + 1}: train loss {loss:.4f} ({step_times[-1]:.2f} s)')
    final = validation_loss(model, val, args.sequence_length, args.val_windows)
    print(f'step {args.steps}: validation loss {final:.4f}')

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    prompt = np.array([enc.encode('ROMEO:')])
    sample = enc.decode(model.generate(prompt, 40, temperature=0.8, top_k=40, rng=rng)[0].tolist())
    print(sample)

    timed = step_times[1:] if len(step_times) > 1 else step_times   # first step includes warm-up
    result = {'config': {k: os.path.relpath(v, ROOT) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'model': 'GPT-2 small (124M), pretrained, NumPy float32',
              'hardware': system_info(),
              'initial_validation_loss': initial, 'final_validation_loss': final,
              'final_validation_perplexity': math.exp(final),
              'time_per_step_median_s': float(np.median(timed)),
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # Linux: KiB
              'train_losses': train_losses, 'step_times_s': step_times, 'sample': sample}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('saved', args.output / 'results.json')


if __name__ == '__main__':
    main()

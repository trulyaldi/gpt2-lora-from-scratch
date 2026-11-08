"""LoRA vs full fine-tuning of pretrained GPT-2 small (124M) on Tiny Shakespeare.

    python benchmark.py                       # both arms, each in its own process, then aggregate
    python benchmark.py --arm lora --steps 20 # one arm only (used internally by the driver)

Each arm runs in a fresh process so CUDA kernel caches, the allocator and peak-memory counters
are not shared between arms. Both arms use the same pretrained weights, the same seeded training
windows, batch shape, step count, schedule shape and validation set; they differ only in which
parameters are trained and in their (recorded) peak learning rates.
"""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
HF_REPO = 'openai-community/gpt2'
ARMS = ('full', 'lora')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=ARMS, help='run a single arm; omit to run both and aggregate')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--sequence-length', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--warmup-steps', type=int, default=20, help='linear LR warmup, then constant')
    p.add_argument('--timing-skip', type=int, default=10, help='steps excluded from time/step stats')
    p.add_argument('--lr-full', type=float, default=3e-5)
    p.add_argument('--lr-lora', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=16)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--data', type=Path, default=ROOT / 'data' / 'tinyshakespeare.txt')
    p.add_argument('--output', type=Path, default=ROOT / 'results' / 'shakespeare_gpt2')
    args = p.parse_args(argv)
    for name in ('steps', 'batch_size', 'sequence_length', 'eval_every'):
        if getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.sequence_length > 1024:
        p.error('GPT-2 context is 1024 tokens')
    if args.timing_skip >= args.steps:
        p.error('--timing-skip must be smaller than --steps')
    return args


# ---------------------------------------------------------------- data
def load_tokens(path):
    import numpy as np
    import tiktoken
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    text = path.read_text(encoding='utf-8')
    tokens = np.array(tiktoken.get_encoding('gpt2').encode_ordinary(text), dtype=np.int64)
    cut = int(0.9 * len(tokens))  # contiguous split: the last 10% of the play text is held out
    return tokens[:cut], tokens[cut:], len(text)


def train_positions(n_train, args):
    import torch
    g = torch.Generator().manual_seed(args.seed)
    return torch.randint(n_train - args.sequence_length, (args.steps, args.batch_size), generator=g)


def windows(data, starts, length, device):
    import torch
    ix = starts[:, None] + torch.arange(length + 1)
    w = data[ix]
    return w[:, :-1].to(device, non_blocking=True), w[:, 1:].to(device, non_blocking=True)


def evaluate(model, valid, args, device):
    """Mean token cross-entropy over every non-overlapping window of the validation split."""
    import torch
    starts = torch.arange(0, len(valid) - args.sequence_length, args.sequence_length)
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for chunk in starts.split(args.batch_size):
            x, y = windows(valid, chunk, args.sequence_length, device)
            total += model(x, y)[1].item() * y.numel()
            count += y.numel()
    model.train()
    return total / count


# ---------------------------------------------------------------- one arm
def run_arm(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from huggingface_hub import hf_hub_download
    from gpt_torch import load_pretrained_gpt2
    from lora import apply_lora

    if not torch.cuda.is_available():
        sys.exit('CUDA is required: peak GPU memory is one of the reported metrics')
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    train_np, valid_np, _ = load_tokens(args.data)
    train, valid = torch.from_numpy(train_np), torch.from_numpy(valid_np)
    positions = train_positions(len(train), args)

    model = load_pretrained_gpt2(hf_hub_download(HF_REPO, 'model.safetensors')).to(device)
    lr = args.lr_full if args.arm == 'full' else args.lr_lora
    targets = None
    if args.arm == 'lora':
        targets = apply_lora(model, args.rank, args.alpha, args.dropout)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup_steps))

    history = [{'step': 0, 'validation_loss': evaluate(model, valid, args, device)}]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()  # peak covers training (incl. periodic eval), not model load
    step_times, train_losses = [], []
    wall = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = windows(train, positions[step - 1], args.sequence_length, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        train_losses.append(loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            history.append({'step': step, 'validation_loss': evaluate(model, valid, args, device),
                            'train_loss_window_mean': statistics.fmean(train_losses[-args.eval_every:])})
    wall = time.perf_counter() - wall

    timed = sorted(step_times[args.timing_skip:])
    tokens_per_step = args.batch_size * args.sequence_length
    median = statistics.median(timed)
    result = {
        'arm': args.arm, 'learning_rate': lr, 'lora_targets': targets,
        'trainable_parameters': trainable, 'total_parameters': total,
        'trainable_percent': 100 * trainable / total,
        'peak_gpu_memory_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_gpu_memory_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        'time_per_step_median_ms': 1000 * median,
        'time_per_step_p10_ms': 1000 * timed[int(0.1 * (len(timed) - 1))],
        'time_per_step_p90_ms': 1000 * timed[int(0.9 * (len(timed) - 1))],
        'timed_steps': len(timed),
        'train_tokens_per_second_median': tokens_per_step / median,
        'wall_seconds_including_eval': wall,
        'initial_validation_loss': history[0]['validation_loss'],
        'final_validation_loss': history[-1]['validation_loss'],
        'final_validation_perplexity': math.exp(history[-1]['validation_loss']),
        'train_losses': train_losses, 'step_times_s': step_times, 'history': history,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'{args.arm}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"{args.arm}: val {result['initial_validation_loss']:.4f} -> {result['final_validation_loss']:.4f}, "
          f"{median * 1000:.1f} ms/step, peak {result['peak_gpu_memory_allocated_mib']:.0f} MiB")


# ---------------------------------------------------------------- environment record
def hardware():
    import torch
    import peft
    info = {'python': platform.python_version(), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'peft': peft.__version__, 'os': platform.platform(),
            'wsl2': 'microsoft' in platform.release().lower()}
    try:
        cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except (OSError, StopIteration):
        cpu = platform.processor()
    info.update(cpu=cpu, cpu_logical_cores=os.cpu_count())
    try:
        mem_kb = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemTotal'))
        info['ram_gib'] = round(mem_kb / 2**20, 1)
    except (OSError, StopIteration):
        pass
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, gpu_memory_mib=round(prop.total_memory / 2**20),
                    gpu_compute_capability=f'{prop.major}.{prop.minor}')
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        info['nvidia_driver'] = out or None
    except (OSError, subprocess.SubprocessError):
        info['nvidia_driver'] = None
    return info


# ---------------------------------------------------------------- aggregate
SERIES = {'full': ('Full fine-tune', '#2a78d6'), 'lora': ('LoRA', '#eb6834')}  # categorical slots 1, 2


def plot(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    plt.rcParams.update({'font.size': 10, 'text.color': ink, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    for name, key, title in [('validation_loss.png', 'validation_loss', 'Validation loss (Tiny Shakespeare, held-out 10%)'),
                             ('train_loss.png', 'train', 'Training loss per step')]:
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
        for arm in ARMS:
            label, color = SERIES[arm]
            r = results[arm]
            if key == 'train':
                xs = list(range(1, len(r['train_losses']) + 1)); ys = r['train_losses']
                ax.plot(xs, ys, color=color, lw=1, alpha=0.35)
                k = 20  # trailing mean for readability; raw per-step values drawn faintly underneath
                sm = [statistics.fmean(ys[max(0, i - k + 1):i + 1]) for i in range(len(ys))]
                ax.plot(xs, sm, color=color, lw=2, label=f'{label} ({k}-step mean)')
                end_x, end_y = xs[-1], sm[-1]
            else:
                xs = [h['step'] for h in r['history']]; ys = [h[key] for h in r['history']]
                ax.plot(xs, ys, color=color, lw=2, marker='o', ms=4, label=label)
                end_x, end_y = xs[-1], ys[-1]
            ax.annotate(f'{label} {end_y:.3f}', (end_x, end_y), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=9, color=ink)
        ax.set_title(title, loc='left', fontsize=11, color=ink)
        ax.set_xlabel('Optimizer step'); ax.set_ylabel('Cross-entropy (nats/token)')
        ax.grid(axis='y', color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(grid)
        ax.margins(x=0.12)
        ax.legend(frameon=False, loc='upper right')
        fig.tight_layout(); fig.savefig(output / name, facecolor=surface); plt.close(fig)


def table(results):
    f, l = results['full'], results['lora']
    rows = [
        ('Trainable parameters', f"{f['trainable_parameters']:,} ({f['trainable_percent']:.1f}%)",
         f"{l['trainable_parameters']:,} ({l['trainable_percent']:.2f}%)"),
        ('Peak GPU memory allocated (MiB)', f"{f['peak_gpu_memory_allocated_mib']:,.0f}", f"{l['peak_gpu_memory_allocated_mib']:,.0f}"),
        ('Peak GPU memory reserved (MiB)', f"{f['peak_gpu_memory_reserved_mib']:,.0f}", f"{l['peak_gpu_memory_reserved_mib']:,.0f}"),
        ('Time/step, median [p10–p90] (ms)',
         f"{f['time_per_step_median_ms']:.0f} [{f['time_per_step_p10_ms']:.0f}–{f['time_per_step_p90_ms']:.0f}]",
         f"{l['time_per_step_median_ms']:.0f} [{l['time_per_step_p10_ms']:.0f}–{l['time_per_step_p90_ms']:.0f}]"),
        ('Validation loss, step 0', f"{f['initial_validation_loss']:.4f}", f"{l['initial_validation_loss']:.4f}"),
        ('Validation loss, final', f"{f['final_validation_loss']:.4f}", f"{l['final_validation_loss']:.4f}"),
        ('Validation perplexity, final', f"{f['final_validation_perplexity']:.2f}", f"{l['final_validation_perplexity']:.2f}"),
        ('Peak learning rate', f"{f['learning_rate']:g}", f"{l['learning_rate']:g}"),
    ]
    return ('| Metric | Full fine-tune | LoRA |\n|---|---:|---:|\n'
            + ''.join(f'| {a} | {b} | {c} |\n' for a, b, c in rows))


def drive(args, argv):
    args.output.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        print(f'== {arm} ==', flush=True)
        subprocess.run([sys.executable, __file__, '--arm', arm, *argv], check=True, cwd=ROOT)
    results = {arm: json.loads((args.output / f'{arm}.json').read_text()) for arm in ARMS}
    train_np, valid_np, chars = load_tokens(args.data)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != 'arm'}
    for k in ('data', 'output'):
        config[k] = os.path.relpath(config[k], ROOT)
    record = {
        'config': config,
        'data': {'name': 'Tiny Shakespeare', 'url': DATA_URL, 'characters': chars, 'tokenizer': 'GPT-2 BPE (tiktoken)',
                 'train_tokens': len(train_np), 'validation_tokens': len(valid_np),
                 'split': 'contiguous 90/10 by token'},
        'model': {'name': 'GPT-2 small (124M)', 'weights': f'huggingface.co/{HF_REPO} model.safetensors'},
        'determinism': 'torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8',
        'hardware': hardware(),
        'results': {arm: {k: v for k, v in r.items() if k not in ('train_losses', 'step_times_s')}
                    for arm, r in results.items()},
    }
    (args.output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    hw = record['hardware']
    md = (f"# LoRA vs full fine-tuning: GPT-2 small on Tiny Shakespeare\n\n{table(results)}\n"
          f"Hardware: {hw.get('gpu')} ({hw.get('gpu_memory_mib')} MiB), driver {hw.get('nvidia_driver')}, "
          f"{hw['cpu']}, {hw.get('ram_gib')} GiB RAM, {'WSL2' if hw['wsl2'] else hw['os']}; "
          f"torch {hw['torch']} (CUDA {hw['torch_cuda']}), peft {hw['peft']}, Python {hw['python']}.\n\n"
          f"Setup: {args.steps} steps, batch {args.batch_size} x {args.sequence_length} tokens, AdamW "
          f"(weight decay {args.weight_decay}), {args.warmup_steps}-step linear warmup then constant LR, grad clip 1.0, "
          f"seed {args.seed}. LoRA rank {args.rank}, alpha {args.alpha:g}, dropout {args.dropout:g} on "
          f"attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj. Validation = all {len(valid_np) // args.sequence_length} "
          f"non-overlapping {args.sequence_length}-token windows of the held-out split. Time/step = synchronized "
          f"forward+backward+clip+optimizer step, first {args.timing_skip} steps and evaluation excluded. Peak memory "
          f"is measured after model load, over training and periodic evaluation.\n")
    (args.output / 'comparison.md').write_text(md)
    plot(results, args.output)
    print(md)


def main():
    argv = sys.argv[1:]
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    if args.arm:
        run_arm(args)
    else:
        drive(args, argv)


if __name__ == '__main__':
    main()

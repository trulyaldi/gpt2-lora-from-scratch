"""One training run of pretrained GPT-2 small (full fine-tuning, LoRA or QLoRA) and checkpoint evaluation.

Every method goes through the same loop: same seeded windows, effective batch, schedule shape,
loss, gradient clipping and evaluation points; they differ only in which parameters train
(and, for QLoRA, in storing the frozen projections in NF4). Training sees only the train and
validation splits; the test split is read only by evaluate_checkpoint().
"""
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

from common import gpt2_weights_path, load_splits, set_seed
from gpt_torch import load_pretrained_gpt2
from lora import apply_lora
from qlora import apply_qlora

METHODS = ('full', 'lora', 'qlora')
MIB = 2 ** 20


def build_model(method, rank=8, alpha=16, dropout=0.0, device='cuda'):
    """Pretrained GPT-2 on device, wrapped for the method. Returns (model, pretrained parameter count)."""
    model = load_pretrained_gpt2(gpt2_weights_path()).to(device)
    pretrained = sum(p.numel() for p in model.parameters())  # before QLoRA turns weights into buffers
    return wrap(model, method, rank, alpha, dropout), pretrained


def wrap(model, method, rank=8, alpha=16, dropout=0.0):
    """Prepare a model for the method: full trains everything, LoRA/QLoRA freeze it and add adapters."""
    if method == 'lora':
        apply_lora(model, rank, alpha, dropout)
    elif method == 'qlora':
        apply_qlora(model, rank, alpha, dropout)
    elif method == 'full':
        model.requires_grad_(True)
    else:
        raise ValueError(f'unknown method {method!r}')
    return model


def trainable_state(model):
    """CPU copies of every trainable parameter, by name (all weights for full, A/B for adapters)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    """Load a trainable_state() into a freshly built model of the same method; names must match exactly."""
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError(f'checkpoint parameters do not match the model: '
                         f'missing {sorted(expected - set(state))[:3]}, extra {sorted(set(state) - expected)[:3]}')
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name, value in state.items():
            params[name].copy_(value)


def batch_positions(seed, steps, batch_size, n_tokens, sequence_length):
    """Start index of every training window, shape (steps, batch_size). Depends only on the seed and
    the data/batch shape, never on the method or micro-batch size, so all methods see the same order."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(n_tokens - sequence_length, (steps, batch_size), generator=g)


def windows(data, starts, length, device):
    """Input/target token windows of the given length starting at each index in starts."""
    ix = starts[:, None] + torch.arange(length + 1)
    w = data[ix]
    return w[:, :-1].to(device, non_blocking=True), w[:, 1:].to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, tokens, sequence_length, batch_size, device):
    """Mean token cross-entropy over every non-overlapping window of a split."""
    starts = torch.arange(0, len(tokens) - sequence_length, sequence_length)
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for chunk in starts.split(batch_size):
        x, y = windows(tokens, chunk, sequence_length, device)
        total += model(x, y)[1].item() * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / count


def accumulate_gradients(model, data, rows, micro_batch_size, sequence_length, device):
    """Backward the mean loss over the windows starting at rows, in micro-batches; returns that mean
    (a 0-d tensor). Windows have equal length, so this equals one backward over the whole batch."""
    total = 0.0
    for chunk in rows.split(micro_batch_size):
        x, y = windows(data, chunk, sequence_length, device)
        loss = model(x, y)[1]
        (loss * (len(chunk) / len(rows))).backward()
        total += loss.detach() * len(chunk)
    return total / len(rows)


def lr_factor(step, warmup, steps, min_ratio):
    """LR multiplier for optimizer step `step` (1-based): linear warmup, then cosine decay to min_ratio.
    min_ratio=1 gives warmup-then-constant (the original 300-step benchmark)."""
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def tensor_mib(tensors):
    return sum(t.numel() * t.element_size() for t in tensors) / MIB


def optimizer_state_mib(optimizer):
    return tensor_mib(t for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t) and t.dim() > 0)


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def train(cfg):
    """Run one training job described by cfg (see benchmark.py) in cfg['run_dir'].

    Writes history.csv (one row per evaluation), steps.csv (one row per optimizer step), summary.json,
    and best.pt (trainable weights at the lowest validation loss) when cfg['save_best'] is set.
    last.pt is written every cfg['checkpoint_every'] steps and resumed from if present."""
    run_dir = Path(cfg['run_dir'])
    if (run_dir / 'summary.json').exists():
        print(f'{run_dir}: already complete, skipping')
        return json.loads((run_dir / 'summary.json').read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; this benchmark measures GPU training only')
    if cfg['batch_size'] % cfg['micro_batch_size']:
        raise ValueError('batch_size must be a multiple of micro_batch_size')
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device, sync = torch.device('cuda'), torch.cuda.synchronize
    set_seed(cfg['seed'])
    torch.use_deterministic_algorithms(cfg['deterministic'])
    free_at_start, physical = (b / MIB for b in torch.cuda.mem_get_info())

    tokens, split_meta = load_splits(('train', 'validation'))
    train_t, val_t = torch.from_numpy(tokens['train']), torch.from_numpy(tokens['validation'])
    steps, batch, micro, length = cfg['steps'], cfg['batch_size'], cfg['micro_batch_size'], cfg['sequence_length']
    positions = batch_positions(cfg['seed'], steps, batch, len(train_t), length)

    model, pretrained = build_model(cfg['method'], cfg['rank'], cfg['alpha'], cfg['dropout'], device)
    sync()
    torch.cuda.empty_cache()
    weights_mib = torch.cuda.memory_allocated() / MIB  # model weights and buffers as stored on the GPU
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s + 1, cfg['warmup_steps'], steps, cfg['min_lr_ratio']))

    state = {'step': 0, 'train_time_s': 0.0, 'history': [], 'steps': [], 'best_val': math.inf,
             'best_step': None, 'best_state': None, 'peak_alloc_mib': 0.0, 'peak_reserved_mib': 0.0,
             'optimizer_state_mib': None, 'resumed_from': None}
    last = run_dir / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=True)
        load_trainable_state(model, ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        schedule.load_state_dict(ck['schedule'])
        torch.set_rng_state(ck['cpu_rng'])
        torch.cuda.set_rng_state(ck['cuda_rng'])
        state = ck['state']
        state['resumed_from'] = state['step']
        print(f'resuming {run_dir} from step {state["step"]}')
    else:
        v = evaluate(model, val_t, length, micro, device)
        state['history'].append({'step': 0, 'tokens': 0, 'train_time_s': 0.0, 'train_loss': math.nan,
                                 'validation_loss': v, 'lr': 0.0, 'peak_allocated_mib': 0.0,
                                 'peak_reserved_mib': 0.0, 'step_time_median_s': math.nan})
        state['best_val'], state['best_step'] = v, 0
        state['best_state'] = trainable_state(model)

    def save_last():
        tmp = last.with_suffix('.tmp')
        torch.save({'model': trainable_state(model), 'optimizer': optimizer.state_dict(),
                    'schedule': schedule.state_dict(), 'cpu_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(), 'state': state}, tmp)
        tmp.replace(last)

    wall = time.perf_counter()
    model.train()
    for step in range(state['step'] + 1, steps + 1):
        lr = optimizer.param_groups[0]['lr']
        sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = accumulate_gradients(model, train_t, positions[step - 1], micro, length, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        optimizer.step()
        schedule.step()
        sync()
        dt = time.perf_counter() - t0
        train_loss = mean_loss.item()
        if not math.isfinite(train_loss):
            raise FloatingPointError(f'non-finite training loss {train_loss} at step {step}')
        peak_a, peak_r = torch.cuda.max_memory_allocated() / MIB, torch.cuda.max_memory_reserved() / MIB
        if peak_r > physical:  # WSL2/Windows can spill to shared system RAM instead of raising OOM
            raise torch.cuda.OutOfMemoryError(
                f'step {step} reserved {peak_r:.0f} MiB, more than the {physical:.0f} MiB of GPU memory: the driver '
                'spilled into shared system memory, so timings are invalid. Treating this as out of memory.')
        if state['optimizer_state_mib'] is None:
            state['optimizer_state_mib'] = optimizer_state_mib(optimizer)
        state['peak_alloc_mib'] = max(state['peak_alloc_mib'], peak_a)
        state['peak_reserved_mib'] = max(state['peak_reserved_mib'], peak_r)
        state['train_time_s'] += dt
        state['step'] = step
        state['steps'].append({'step': step, 'tokens': step * batch * length, 'train_loss': train_loss, 'lr': lr,
                               'grad_norm': grad_norm.item(), 'step_time_s': dt,
                               'train_time_s': state['train_time_s'],
                               'peak_allocated_mib': peak_a, 'peak_reserved_mib': peak_r})

        if step % cfg['eval_every'] == 0 or step == steps:
            v = evaluate(model, val_t, length, micro, device)
            recent = state['steps'][-cfg['eval_every']:]
            state['history'].append({'step': step, 'tokens': step * batch * length,
                                     'train_time_s': state['train_time_s'],
                                     'train_loss': statistics.fmean(r['train_loss'] for r in recent),
                                     'validation_loss': v, 'lr': lr,
                                     'peak_allocated_mib': state['peak_alloc_mib'],
                                     'peak_reserved_mib': state['peak_reserved_mib'],
                                     'step_time_median_s': statistics.median(r['step_time_s'] for r in recent)})
            if v < state['best_val']:
                state['best_val'], state['best_step'] = v, step
                state['best_state'] = trainable_state(model)
            print(f"{cfg['method']} lr={cfg['lr']:g} seed={cfg['seed']} step {step}/{steps}: "
                  f"train {state['history'][-1]['train_loss']:.4f} val {v:.4f} "
                  f"({state['train_time_s']:.0f}s train)", flush=True)
        if cfg['checkpoint_every'] and step % cfg['checkpoint_every'] == 0 and step < steps:
            save_last()
    wall = time.perf_counter() - wall

    if cfg['save_best']:
        torch.save({'method': cfg['method'], 'step': state['best_step'], 'validation_loss': state['best_val'],
                    'state': state['best_state']}, run_dir / 'best.pt')
    write_csv(run_dir / 'history.csv', state['history'])
    write_csv(run_dir / 'steps.csv', state['steps'])

    times = sorted(r['step_time_s'] for r in state['steps'][cfg['timing_skip']:]) or [math.nan]
    best_row = next(h for h in state['history'] if h['step'] == state['best_step'])
    grads_mib = tensor_mib(params)
    summary = {
        'status': 'completed', 'config': cfg,
        'data': {k: v for k, v in split_meta['splits'].items() if k in ('train', 'validation')},
        'trainable_parameters': trainable, 'pretrained_parameters': pretrained,
        'trainable_percent': 100 * trainable / (pretrained + (0 if cfg['method'] == 'full' else trainable)),
        'memory_mib': {
            'model_after_load': weights_mib, 'trainable_parameters': grads_mib, 'gradients': grads_mib,
            'optimizer_state': state['optimizer_state_mib'],
            'peak_allocated': state['peak_alloc_mib'], 'peak_reserved': state['peak_reserved_mib'],
            'gpu_total': physical, 'gpu_free_at_start': free_at_start,
            'activations_and_other_at_peak': state['peak_alloc_mib'] - weights_mib - grads_mib
                                             - state['optimizer_state_mib'],
        },
        'time': {'train_s': state['train_time_s'], 'train_s_to_best': best_row['train_time_s'],
                 'wall_s_this_process_incl_eval': wall,
                 'step_median_ms': 1000 * statistics.median(times),
                 'step_p10_ms': 1000 * times[int(0.1 * (len(times) - 1))],
                 'step_p90_ms': 1000 * times[int(0.9 * (len(times) - 1))]},
        'tokens_total': steps * batch * length, 'tokens_to_best': best_row['tokens'],
        'initial_validation_loss': state['history'][0]['validation_loss'],
        'final_validation_loss': state['history'][-1]['validation_loss'],
        'best_validation_loss': state['best_val'], 'best_step': state['best_step'],
        'resumed_from_step': state['resumed_from'],
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    last.unlink(missing_ok=True)
    return summary


def evaluate_checkpoint(run_dir, device='cuda'):
    """Reload best.pt of a finished run, re-check its validation loss against the recorded value,
    and measure test loss. Only called after hyperparameter selection, on the final runs."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text())
    cfg = summary['config']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(cfg['deterministic'])
    model, _ = build_model(cfg['method'], cfg['rank'], cfg['alpha'], cfg['dropout'], device)
    ck = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    load_trainable_state(model, ck['state'])
    tokens, _ = load_splits(('validation', 'test'))
    length, batch = cfg['sequence_length'], cfg['micro_batch_size']
    val = evaluate(model, torch.from_numpy(tokens['validation']), length, batch, device)
    if abs(val - ck['validation_loss']) > 1e-4:
        raise RuntimeError(f'{run_dir}: reloaded checkpoint gives validation loss {val:.6f}, '
                           f'recorded {ck["validation_loss"]:.6f}')
    test = evaluate(model, torch.from_numpy(tokens['test']), length, batch, device)
    result = {'checkpoint_step': ck['step'], 'selected_by': 'lowest validation loss',
              'recorded_validation_loss': ck['validation_loss'], 'reloaded_validation_loss': val,
              'test_loss': test, 'test_perplexity': math.exp(test)}
    (run_dir / 'test.json').write_text(json.dumps(result, indent=2) + '\n')
    del model
    torch.cuda.empty_cache()
    return result

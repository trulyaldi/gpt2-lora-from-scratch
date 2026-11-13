"""Aggregate final runs across seeds and write CSV/JSON summaries, figures and BENCHMARK_REPORT.md.

Reads only files written by training.py and the test stage; never trains or evaluates a model.
"""
import csv
import json
import math
import os
import statistics
from pathlib import Path

METHODS = ('full', 'lora', 'qlora')
SERIES = {'full': ('Full fine-tune', '#2a78d6', 'o'), 'lora': ('LoRA (r=8)', '#eb6834', 's'),
          'qlora': ('QLoRA (r=8, NF4)', '#1baf7a', '^')}


def mean_std(values):
    """(mean, sample standard deviation with ddof=1 (nan for n<2), n) of the finite values."""
    v = [x for x in values if x is not None and math.isfinite(x)]
    if not v:
        return math.nan, math.nan, 0
    return statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else math.nan, len(v)


def read_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_runs(output):
    """{'final': {method: [run, ...]}, 'failed': [...], 'selected': ..., 'environment': ...}."""
    output = Path(output)
    runs, failed = {m: [] for m in METHODS}, []
    for method in METHODS:
        for run_dir in sorted((output / 'runs' / 'final' / method).glob('seed*')):
            if (run_dir / 'failed.json').exists() or not (run_dir / 'summary.json').exists():
                failed.append({'run': os.path.relpath(run_dir, output), 'stage': 'final',
                               **(json.loads((run_dir / 'failed.json').read_text())
                                  if (run_dir / 'failed.json').exists() else {'returncode': 'incomplete'})})
                continue
            run = json.loads((run_dir / 'summary.json').read_text())
            run['history'] = read_csv(run_dir / 'history.csv')
            run['test'] = json.loads((run_dir / 'test.json').read_text()) if (run_dir / 'test.json').exists() else None
            run['seed'] = run['config']['seed']
            runs[method].append(run)
    for path in sorted((output / 'runs' / 'tune').glob('*/*/failed.json')):
        failed.append({'run': os.path.relpath(path.parent, output), 'stage': 'tune', **json.loads(path.read_text())})
    read = lambda name: json.loads((output / name).read_text()) if (output / name).exists() else None
    return {'final': runs, 'failed': failed, 'selected': read('selected_hparams.json'),
            'environment': read('environment.json')}


METRICS = {  # name -> (label, getter)
    'test_loss': ('Test loss (nats/token)', lambda r: r['test']['test_loss'] if r['test'] else None),
    'test_perplexity': ('Test perplexity', lambda r: r['test']['test_perplexity'] if r['test'] else None),
    'best_validation_loss': ('Best validation loss', lambda r: r['best_validation_loss']),
    'initial_validation_loss': ('Validation loss before training', lambda r: r['initial_validation_loss']),
    'best_step': ('Selected checkpoint step', lambda r: r['best_step']),
    'train_s': ('Total training time (s)', lambda r: r['time']['train_s']),
    'train_s_to_best': ('Training time to selected checkpoint (s)', lambda r: r['time']['train_s_to_best']),
    'step_median_ms': ('Time per step, median (ms)', lambda r: r['time']['step_median_ms']),
    'model_after_load_mib': ('Model memory after load (MiB)', lambda r: r['memory_mib']['model_after_load']),
    'gradients_mib': ('Gradient memory (MiB)', lambda r: r['memory_mib']['gradients']),
    'optimizer_state_mib': ('AdamW state memory (MiB)', lambda r: r['memory_mib']['optimizer_state']),
    'activations_and_other_mib': ('Activations and other at peak (MiB)',
                                  lambda r: r['memory_mib']['activations_and_other_at_peak']),
    'peak_allocated_mib': ('Peak GPU memory allocated (MiB)', lambda r: r['memory_mib']['peak_allocated']),
    'peak_reserved_mib': ('Peak GPU memory reserved (MiB)', lambda r: r['memory_mib']['peak_reserved']),
    'trainable_parameters': ('Trainable parameters', lambda r: r['trainable_parameters']),
}


def aggregate(runs):
    """{method: {metric: {'mean', 'std', 'n'}}} over seeds."""
    out = {}
    for method, rs in runs.items():
        if rs:
            out[method] = {k: dict(zip(('mean', 'std', 'n'), mean_std([get(r) for r in rs])))
                           for k, (_, get) in METRICS.items()}
    return out


def write_tables(output, data, agg):
    """final_runs.csv (one row per seed), summary.csv/json (mean, std, n per method), tuning.csv."""
    rows = [{'method': m, 'seed': r['seed'], 'lr': r['config']['lr'], **{k: get(r) for k, (_, get) in METRICS.items()}}
            for m, rs in data['final'].items() for r in rs]
    if rows:
        with open(output / 'final_runs.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    with open(output / 'summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'metric', 'mean', 'std', 'n'])
        for m, metrics in agg.items():
            for k, s in metrics.items():
                w.writerow([m, k, s['mean'], s['std'], s['n']])
    (output / 'summary.json').write_text(json.dumps(agg, indent=2) + '\n')
    if data['selected']:
        with open(output / 'tuning.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'lr', 'status', 'best_validation_loss', 'best_step', 'selected'])
            for m, s in data['selected'].items():
                for c in s['candidates']:
                    w.writerow([m, c['lr'], c['status'], c.get('best_validation_loss'), c.get('best_step'),
                                c['lr'] == s['lr']])


# ---------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'text.color': '#0b0b0b', 'axes.labelcolor': '#52514e',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.facecolor': '#fcfcfb',
                         'figure.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'axes.edgecolor': '#e4e3df',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'axes.axisbelow': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.8,
                         'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
    return plt


def _title(ax, text):
    ax.set_title(text, loc='left', fontsize=10.5, color='#0b0b0b')


def plot_curves(runs, figures, x_key, x_label, filename, title, x_scale=1.0):
    """Validation loss against x: each seed as a thin transparent line, the seed mean as a strong line.
    Seeds share evaluation steps, so the mean is taken per evaluation (x averaged too, for time)."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for method in METHODS:
        if not runs[method]:
            continue
        label, color, marker = SERIES[method]
        n = min(len(r['history']) for r in runs[method])
        for r in runs[method]:
            ax.plot([h[x_key] * x_scale for h in r['history']], [h['validation_loss'] for h in r['history']],
                    color=color, lw=1, alpha=0.3)
        xs = [statistics.fmean(r['history'][i][x_key] for r in runs[method]) * x_scale for i in range(n)]
        ys = [statistics.fmean(r['history'][i]['validation_loss'] for r in runs[method]) for i in range(n)]
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=4, markevery=max(1, n // 12),
                label=f'{label} (mean of {len(runs[method])} seeds)')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Validation cross-entropy (nats/token)')
    _title(ax, title)
    ax.legend(loc='upper right')
    fig.savefig(figures / filename)
    plt.close(fig)


def plot_memory(agg, figures):
    """Grouped bars of the memory components, mean over seeds, coloured by method."""
    plt = _style()
    groups = [('model_after_load_mib', 'Model\nafter load'), ('gradients_mib', 'Gradients'),
              ('optimizer_state_mib', 'AdamW\nstate'), ('peak_allocated_mib', 'Peak\nallocated'),
              ('peak_reserved_mib', 'Peak\nreserved')]
    methods = [m for m in METHODS if m in agg]
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, m in enumerate(methods):
        label, color, _ = SERIES[m]
        xs = [g + (i - (len(methods) - 1) / 2) * width for g in range(len(groups))]
        ys = [agg[m][k]['mean'] for k, _ in groups]
        bars = ax.bar(xs, ys, width * 0.92, color=color, label=label)
        ax.bar_label(bars, labels=[f'{y:,.0f}' for y in ys], fontsize=7, color='#52514e', padding=2)
    ax.set_xticks(range(len(groups)), [g for _, g in groups])
    ax.set_ylabel('GPU memory (MiB)')
    ax.grid(axis='x', visible=False)
    _title(ax, 'GPU memory by component (mean over seeds)')
    ax.legend(loc='upper left')
    fig.savefig(figures / 'peak_memory.png')
    plt.close(fig)


def plot_test_loss(runs, agg, figures):
    """Mean test loss with +-1 sample std error bars, and every seed as a point."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    methods = [m for m in METHODS if m in agg and agg[m]['test_loss']['n']]
    for i, m in enumerate(methods):
        label, color, marker = SERIES[m]
        s = agg[m]['test_loss']
        seeds = [r['test']['test_loss'] for r in runs[m] if r['test']]
        offsets = [(j - (len(seeds) - 1) / 2) * 0.06 for j in range(len(seeds))]
        ax.scatter([i + 0.18 + o for o in offsets], seeds, color=color, alpha=0.45, s=18, marker=marker, lw=0)
        ax.errorbar(i, s['mean'], yerr=0 if math.isnan(s['std']) else s['std'], color=color, marker=marker,
                    ms=8, capsize=6, lw=2)
        ax.annotate(f"{s['mean']:.3f}" + ('' if math.isnan(s['std']) else f" ± {s['std']:.3f}"), (i, s['mean']),
                    xytext=(-12, 0), textcoords='offset points', ha='right', va='center', fontsize=8.5)
    ax.set_xticks(range(len(methods)), [SERIES[m][0] for m in methods])
    ax.set_xlim(-0.8, len(methods) - 0.4)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('Test cross-entropy (nats/token)')
    _title(ax, 'Test loss at the best-validation checkpoint (mean ± 1 std; dots = seeds)')
    fig.savefig(figures / 'test_loss.png')
    plt.close(fig)


def plot_tradeoffs(agg, figures):
    """Test loss against total training time and against peak memory (mean ± std on both axes)."""
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, key, xlabel in ((axes[0], 'train_s_to_best', 'Training time to selected checkpoint (s)'),
                            (axes[1], 'peak_allocated_mib', 'Peak GPU memory allocated (MiB)')):
        for m in METHODS:
            if m not in agg or not agg[m]['test_loss']['n']:
                continue
            label, color, marker = SERIES[m]
            x, y = agg[m][key], agg[m]['test_loss']
            ax.errorbar(x['mean'], y['mean'], xerr=None if math.isnan(x['std']) else x['std'],
                        yerr=None if math.isnan(y['std']) else y['std'], color=color, marker=marker, ms=8,
                        capsize=4, lw=1.5, label=label)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel('Test cross-entropy (nats/token)')
    _title(axes[0], 'Quality vs training time')
    _title(axes[1], 'Quality vs peak memory')
    axes[1].legend(loc='upper right')
    fig.savefig(figures / 'tradeoffs.png')
    plt.close(fig)


# ---------------------------------------------------------------- report
def fmt(s, digits=3, comma=False):
    if not s['n']:
        return 'n/a'
    f = (lambda v: f'{v:,.0f}') if comma else (lambda v: f'{v:.{digits}f}')
    return f(s['mean']) + ('' if math.isnan(s['std']) else f" ± {f(s['std'])}")


def pareto(agg, keys=('test_loss', 'train_s_to_best', 'peak_allocated_mib')):
    """Methods not dominated on the given lower-is-better metrics (mean values)."""
    ms = [m for m in agg if agg[m]['test_loss']['n']]
    def dominates(a, b):
        va, vb = [agg[a][k]['mean'] for k in keys], [agg[b][k]['mean'] for k in keys]
        return all(x <= y for x, y in zip(va, vb)) and any(x < y for x, y in zip(va, vb))
    return [m for m in ms if not any(dominates(o, m) for o in ms if o != m)]


def comparison_text(agg):
    """Measured differences relative to full fine-tuning, with a flag when a quality gap is
    within seed-to-seed variation (|difference of means| < the larger of the two std)."""
    lines = []
    if 'full' not in agg or not agg['full']['test_loss']['n']:
        return ['Full fine-tuning has no tested runs, so no relative comparison is made.']
    base = agg['full']
    for m in ('lora', 'qlora'):
        if m not in agg or not agg[m]['test_loss']['n']:
            continue
        a = agg[m]
        d = a['test_loss']['mean'] - base['test_loss']['mean']
        spread = max(a['test_loss']['std'], base['test_loss']['std'])
        noise = '' if math.isnan(spread) else (' (within seed-to-seed std)' if abs(d) < spread
                                               else ' (larger than seed-to-seed std)')
        lines.append(f"- **{SERIES[m][0]} vs full fine-tuning:** test loss {d:+.3f} nats{noise}; "
                     f"training time to the selected checkpoint ×{a['train_s_to_best']['mean'] / base['train_s_to_best']['mean']:.2f}, "
                     f"time per step ×{a['step_median_ms']['mean'] / base['step_median_ms']['mean']:.2f}; "
                     f"peak allocated memory ×{a['peak_allocated_mib']['mean'] / base['peak_allocated_mib']['mean']:.2f} "
                     f"({a['peak_allocated_mib']['mean'] - base['peak_allocated_mib']['mean']:+,.0f} MiB).")
    return lines


def write_report(output, report_path, smoke=False):
    """Write tables, figures and the Markdown report for the results in output."""
    output, report_path = Path(output), Path(report_path)
    data = load_runs(output)
    env = data['environment'] or {}
    cfg, hw = env.get('config', {}), env.get('hardware', {})
    agg = aggregate(data['final'])
    write_tables(output, data, agg)
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    runs = data['final']
    if any(runs.values()):
        plot_curves(runs, figures, 'step', 'Optimizer step', 'validation_vs_steps.png', 'Validation loss vs training steps')
        plot_curves(runs, figures, 'train_time_s', 'Cumulative training time (s, excludes evaluation)',
                    'validation_vs_time.png', 'Validation loss vs training time')
        plot_curves(runs, figures, 'tokens', 'Training tokens processed (millions)', 'validation_vs_tokens.png',
                    'Validation loss vs tokens processed', x_scale=1e-6)
        plot_memory(agg, figures)
        plot_test_loss(runs, agg, figures)
        plot_tradeoffs(agg, figures)

    rel = lambda p: os.path.relpath(p, report_path.parent)
    splits = env.get('data', {}).get('splits', {})
    L = ['# Benchmark report: full fine-tuning vs LoRA vs QLoRA on GPT-2 small', '']
    if smoke:
        L += ['> **SMOKE TEST — NOT BENCHMARK RESULTS.** These numbers come from a deliberately tiny run '
              f"({cfg.get('steps')} steps, {len(cfg.get('seeds', []))} seeds, {len(cfg.get('lr_grid_full', []))} "
              'learning rates per method). It checks that the pipeline works end to end; it says nothing about '
              'which method is better. Run `python benchmark.py all` for the real benchmark.', '']
    L += ['## Experimental setup', '',
          f"- **Model:** pretrained GPT-2 small (124M), {env.get('model', {}).get('weights', '')}; fp32 compute for every method.",
          f"- **Data:** Tiny Shakespeare ({env.get('data', {}).get('characters', 0):,} characters, sha256 "
          f"`{env.get('data', {}).get('sha256', '')[:12]}…`), raw text cut into contiguous 80/10/10 sections at blank "
          'lines, each section tokenized separately with the GPT-2 BPE. Training windows are sampled inside the '
          'training section only, so none crosses a split boundary.',
          '', '| Split | Characters | Tokens |', '|---|---:|---:|']
    L += [f"| {n} | {s['characters']:,} | {s['tokens']:,} |" for n, s in splits.items()]
    eff = cfg.get('batch_size', 0) * cfg.get('sequence_length', 0)
    L += ['', f"- **Training (identical for every method):** {cfg.get('steps')} optimizer steps, effective batch "
          f"{cfg.get('batch_size')} × {cfg.get('sequence_length')} tokens ({cfg.get('micro_batch_size')} per micro-batch, "
          f"gradient accumulation), {eff * cfg.get('steps', 0) / 1e6:.2f}M training tokens "
          f"(≈{eff * cfg.get('steps', 0) / max(1, splits.get('train', {}).get('tokens', 1)):.1f} passes over the training split); "
          f"AdamW, weight decay {cfg.get('weight_decay')}, gradient clipping {cfg.get('grad_clip')}, "
          f"{cfg.get('warmup_steps')}-step linear warmup then cosine decay to {cfg.get('min_lr_ratio')} × peak LR; "
          f"validation every {cfg.get('eval_every')} steps on all non-overlapping {cfg.get('sequence_length')}-token "
          'windows of the validation split. Training-window positions depend only on the seed, so every method sees '
          'the same batches in the same order for a given seed.',
          f"- **LoRA / QLoRA:** rank {cfg.get('rank')}, alpha {cfg.get('alpha')}, no dropout, on attn.c_attn, attn.c_proj, "
          'mlp.c_fc and mlp.c_proj of all 12 blocks. QLoRA stores those frozen projection weights as NF4 (block 64, '
          'fp32 absmax, no double quantization) and dequantizes them on the fly; everything else is as in LoRA.',
          f"- **Learning-rate search:** {len(cfg.get('lr_grid_full', []))} candidates per method, full fine-tuning "
          f"{cfg.get('lr_grid_full')}, LoRA and QLoRA {cfg.get('lr_grid_adapter')} (each tuned separately), one run per "
          f"candidate with tuning seed {cfg.get('tune_seed')} and the same schedule as the final runs. The LR whose "
          'best checkpoint has the lowest validation loss is selected. The test split is not loaded during tuning '
          'or training.',
          f"- **Final runs:** seeds {cfg.get('seeds')} (disjoint from the tuning seed) with the selected LR. For each run the "
          'checkpoint with the lowest validation loss is kept, reloaded (its validation loss is re-measured and must '
          'match), and only then evaluated on the test split.',
          f"- **Measurement:** training time is the sum of per-step times bracketed by `torch.cuda.synchronize()` "
          f"(evaluation and checkpoint I/O excluded); time/step medians skip the first {cfg.get('timing_skip')} steps. "
          'Peak memory is the maximum over training steps of `torch.cuda.max_memory_allocated/reserved()`, reset before '
          'each step. Gradient and optimizer memory are the byte sizes of those tensors; "activations and other" is the '
          'peak minus model, gradients and optimizer state. Each run is its own process.',
          f"- **Reproducibility:** {env.get('reproducibility', {}).get('note', '')} "
          f"Deterministic algorithms: {env.get('reproducibility', {}).get('deterministic_algorithms')}.",
          f"- **Hardware/software:** {hw.get('gpu')} ({hw.get('gpu_memory_mib')} MiB), driver {hw.get('nvidia_driver')}, "
          f"{hw.get('cpu')}, {'WSL2' if hw.get('wsl2') else hw.get('os')}; PyTorch {hw.get('torch')} (CUDA "
          f"{hw.get('torch_cuda')}), Python {hw.get('python')}.", '']

    if data['selected']:
        L += ['## Hyperparameter selection (validation loss, tuning seed)', '',
              '| Method | Learning rate | Best validation loss | At step | Selected |', '|---|---:|---:|---:|:---:|']
        for m, s in data['selected'].items():
            for c in s['candidates']:
                v = f"{c['best_validation_loss']:.4f}" if c['status'] == 'completed' else 'failed'
                L.append(f"| {SERIES[m][0]} | {c['lr']:g} | {v} | {c.get('best_step', '')} | "
                         f"{'✓' if c['lr'] == s['lr'] else ''} |")
        L.append('')

    present = [m for m in METHODS if m in agg]
    n = {m: agg[m]['test_loss']['n'] for m in present}
    if data['selected']:
        edges = [f"{SERIES[m][0]} ({s['lr']:g})" for m, s in data['selected'].items()
                 if len(s['candidates']) > 1 and s['lr'] in (min(c['lr'] for c in s['candidates']),
                                                             max(c['lr'] for c in s['candidates']))]
        if edges:
            L += [f"**Selected learning rate at the edge of its grid:** {', '.join(edges)}. The optimum may lie "
                  'outside the searched range; widen the grid for that method before drawing conclusions.', '']
    L += ['## Results (mean ± sample std over seeds)', '',
          'n = ' + ', '.join(f'{SERIES[m][0]}: {n[m]}' for m in present), '',
          '| Metric | ' + ' | '.join(SERIES[m][0] for m in present) + ' |', '|---|' + '---:|' * len(present)]
    rows = [('test_loss', 3, False), ('test_perplexity', 2, False), ('best_validation_loss', 3, False),
            ('initial_validation_loss', 3, False), ('best_step', 0, False), ('train_s', 0, False),
            ('train_s_to_best', 0, False), ('step_median_ms', 0, False), ('trainable_parameters', 0, True),
            ('model_after_load_mib', 0, True), ('gradients_mib', 0, True), ('optimizer_state_mib', 0, True),
            ('activations_and_other_mib', 0, True), ('peak_allocated_mib', 0, True), ('peak_reserved_mib', 0, True)]
    for key, digits, comma in rows:
        L.append(f'| {METRICS[key][0]} | ' + ' | '.join(fmt(agg[m][key], digits, comma) for m in present) + ' |')
    L += ['', 'Per-seed values: `' + rel(output / 'final_runs.csv') + '`; per-evaluation curves: '
          '`runs/final/<method>/seed<seed>/history.csv`; per-step logs: `steps.csv`.', '']
    tight = sorted({SERIES[m][0] for m, rs in runs.items() for r in rs
                    if r['memory_mib'].get('gpu_free_at_start') and
                    r['memory_mib']['peak_reserved'] > r['memory_mib']['gpu_free_at_start']})
    if tight:
        L += [f"**Memory note:** for {', '.join(tight)}, peak reserved memory exceeded the GPU memory that was free "
              'when the run started (other processes, e.g. the desktop, hold some VRAM). Runs that exceed the '
              "GPU's total memory are rejected as out-of-memory, but a partial spill below that limit cannot be "
              'ruled out; compare its time per step across seeds for outliers.', '']
    if data['failed']:
        L += ['**Failed or incomplete runs (excluded from the statistics above):**', '']
        L += [f"- `{f['run']}` ({f['stage']}): exit {f['returncode']}"
              + (f" — `{f['stderr_tail'][-1]}`" if f.get('stderr_tail') else '') for f in data['failed']]
        L.append('')

    L += ['## Figures', '']
    for name, caption in (('validation_vs_steps.png', 'Validation loss vs training steps'),
                          ('validation_vs_time.png', 'Validation loss vs cumulative training time'),
                          ('validation_vs_tokens.png', 'Validation loss vs tokens processed'),
                          ('peak_memory.png', 'GPU memory by component'),
                          ('test_loss.png', 'Test loss with error bars'),
                          ('tradeoffs.png', 'Quality vs time and quality vs memory')):
        if (figures / name).exists():
            L += [f'![{caption}]({rel(figures / name)})', '']

    L += ['## Quality vs training time and memory', ''] + comparison_text(agg) + ['']
    front = pareto(agg)
    L += ['## Which method gives the best trade-off on a 4 GB GPU?', '']
    if smoke:
        L += ['Not answerable from this smoke test: the runs are far too short and too few to compare quality. '
              'The measured memory and per-step time above are real but only illustrate the pipeline.', '']
    elif front:
        best_q = min(front, key=lambda m: agg[m]['test_loss']['mean'])
        names = ', '.join(SERIES[m][0] for m in front)
        L += [f"Measured Pareto front over (test loss, training time to the selected checkpoint, peak allocated memory), "
              f"all lower-is-better, using seed means: **{names}**. "
              + ('This method is not dominated by the others on any of the three metrics.' if len(front) == 1 else
                 'No single method wins on all three; the choice depends on which constraint binds.')
              + f" Lowest mean test loss among them: {SERIES[best_q][0]}.",
              '', '_Interpretation: see the comparison section above; differences flagged "within seed-to-seed std" '
              'should not be read as a quality ranking._', '']
    L += ['## Limitations and possible confounds', '',
          f"- {min(n.values()) if n else 0} seeds per method and one learning-rate grid of "
          f"{len(cfg.get('lr_grid_full', []))} points; the selected LR may sit at a grid edge (check the table above). "
          'Only the learning rate was tuned (rank, alpha, weight decay, schedule were fixed).',
          '- Learning rates differ between methods by design (each method is tuned on its own grid).',
          '- Weight decay applies to every trainable tensor: for full fine-tuning that includes embeddings, '
          'LayerNorms and biases; for LoRA/QLoRA only the adapter matrices.',
          '- QLoRA starts from a higher validation loss because NF4 quantization perturbs the frozen weights.',
          '- QLoRA dequantizes with ordinary PyTorch ops in every forward and backward pass (no fused kernels like '
          'bitsandbytes), so its per-step time is an upper bound for an optimized implementation.',
          '- Gradient accumulation keeps full-precision gradients alive across micro-batches, which raises full '
          "fine-tuning's peak memory relative to a single large batch; the micro-batch size is limited by fitting "
          'full fine-tuning into 4 GB, so larger batches (where adapter savings grow) were not measured.',
          '- Deterministic algorithms are on for every method. Their cost is not re-measured by this run; one 30-step '
          'check on an RTX 3050 Ti Laptop GPU found 1.6–3.6% per step (full 1.8%, LoRA 1.6%, QLoRA 3.6%).',
          '- One laptop GPU under WSL2; timings depend on clocks, thermals and other GPU load.',
          '- The test split is the last 10% of the text (mostly later plays), so it differs in content from training.', '']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(L))
    print(f'wrote {report_path} and {figures}/')

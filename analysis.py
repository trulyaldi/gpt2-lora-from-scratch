"""Aggregate benchmark runs: mean ± std over seeds, CSV/JSON tables, figures, and the results block in README.md.

Reads only what training.py and the test stage saved, so everything here can be regenerated from results/.
"""
import csv
import json
import math
import os
import statistics
from pathlib import Path

METHODS = ('full', 'lora', 'qlora')
SERIES = {'full': ('Full fine-tune', '#2a78d6', 'o'), 'lora': ('LoRA r=8', '#eb6834', 's'),
          'qlora': ('QLoRA r=8 (NF4)', '#1baf7a', '^')}
START, END = '<!-- results:start -->', '<!-- results:end -->'

METRICS = {  # key -> (table label, getter, decimals)
    'best_validation_loss': ('Validation loss (selected checkpoint)', lambda r: r['best_validation_loss'], 3),
    'test_loss': ('Test loss', lambda r: r['test']['test_loss'] if r['test'] else None, 3),
    'test_perplexity': ('Test perplexity', lambda r: r['test']['test_perplexity'] if r['test'] else None, 2),
    'best_step': ('Selected step', lambda r: r['best_step'], 0),
    'train_s_to_best': ('Training time to selected step (s)', lambda r: r['time']['train_s_to_best'], 0),
    'train_s': ('Total training time (s)', lambda r: r['time']['train_s'], 0),
    'step_median_ms': ('Time per step, median (ms)', lambda r: r['time']['step_median_ms'], 0),
    'model_after_load_mib': ('Model on GPU (MiB)', lambda r: r['memory_mib']['model_after_load'], 0),
    'optimizer_state_mib': ('AdamW state (MiB)', lambda r: r['memory_mib']['optimizer_state'], 0),
    'peak_allocated_mib': ('Peak memory allocated (MiB)', lambda r: r['memory_mib']['peak_allocated'], 0),
    'peak_reserved_mib': ('Peak memory reserved (MiB)', lambda r: r['memory_mib']['peak_reserved'], 0),
    'trainable_parameters': ('Trainable parameters', lambda r: r['trainable_parameters'], 0),
}


def mean_std(values):
    """(mean, sample std with ddof=1 (nan for n < 2), n) over the finite values."""
    v = [x for x in values if x is not None and math.isfinite(x)]
    if not v:
        return math.nan, math.nan, 0
    return statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else math.nan, len(v)


def read_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load(output):
    """Completed final runs per method, failed/incomplete runs, and the LR selection (or None)."""
    runs, failed = {m: [] for m in METHODS}, []
    for stage in ('tune', 'final'):
        for path in sorted((output / 'runs' / stage).glob('*/*/config.json')):
            run_dir = path.parent
            if not (run_dir / 'summary.json').exists():
                failed.append(os.path.relpath(run_dir, output))
            elif stage == 'final':
                run = json.loads((run_dir / 'summary.json').read_text())
                run['history'] = read_csv(run_dir / 'history.csv')
                test = run_dir / 'test.json'
                run['test'] = json.loads(test.read_text()) if test.exists() else None
                runs[run['config']['method']].append(run)
    selected = output / 'selected_hparams.json'
    return runs, failed, json.loads(selected.read_text()) if selected.exists() else None


def aggregate(runs):
    return {m: {k: dict(zip(('mean', 'std', 'n'), mean_std([get(r) for r in rs])))
                for k, (_, get, _) in METRICS.items()}
            for m, rs in runs.items() if rs}


def write_tables(output, runs, agg, selected):
    rows = [{'method': m, 'seed': r['config']['seed'], 'lr': r['config']['lr'],
             **{k: get(r) for k, (_, get, _) in METRICS.items()}} for m, rs in runs.items() for r in rs]
    if rows:
        with open(output / 'final_runs.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (output / 'summary.json').write_text(json.dumps(agg, indent=2) + '\n')
    if selected:
        with open(output / 'tuning.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'lr', 'best_validation_loss', 'best_step', 'selected'])
            for m, s in selected.items():
                for c in s['candidates']:
                    w.writerow([m, c['lr'], c.get('best_validation_loss'), c.get('best_step'), c['lr'] == s['lr']])


# ---------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e',
                         'ytick.color': '#52514e', 'axes.facecolor': '#fcfcfb', 'figure.facecolor': '#fcfcfb',
                         'savefig.facecolor': '#fcfcfb', 'axes.edgecolor': '#d4d3cf', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'axes.axisbelow': True,
                         'grid.color': '#e4e3df', 'legend.frameon': False, 'figure.dpi': 150,
                         'savefig.bbox': 'tight', 'axes.titlelocation': 'left', 'axes.titlesize': 10.5})
    return plt


def plot_curves(runs, path, x_key, x_label, title):
    """Validation loss vs x: seeds as thin transparent lines, their mean (per evaluation) as a strong line."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for m, rs in runs.items():
        if not rs:
            continue
        label, color, marker = SERIES[m]
        for r in rs:
            ax.plot([h[x_key] for h in r['history']], [h['validation_loss'] for h in r['history']],
                    color=color, lw=1, alpha=0.3)
        n = min(len(r['history']) for r in rs)
        mean = lambda key, i: statistics.fmean(r['history'][i][key] for r in rs)
        ax.plot([mean(x_key, i) for i in range(n)], [mean('validation_loss', i) for i in range(n)], color=color,
                lw=2, marker=marker, ms=4, markevery=max(1, n // 10), label=f'{label} (mean of {len(rs)} seeds)')
    ax.set(xlabel=x_label, ylabel='Validation loss (nats/token)', title=title)
    ax.legend(loc='upper right')
    fig.savefig(path)
    plt.close(fig)


def plot_lr_search(selected, path):
    """Best validation loss of every learning-rate candidate (tuning seed); the selected one is ringed."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, s in selected.items():
        label, color, marker = SERIES[m]
        done = [c for c in s['candidates'] if c['status'] == 'completed']
        ax.plot([c['lr'] for c in done], [c['best_validation_loss'] for c in done], color=color, lw=1.5,
                marker=marker, ms=6, label=label)
        best = next(c for c in done if c['lr'] == s['lr'])
        ax.scatter(best['lr'], best['best_validation_loss'], s=180, facecolors='none', edgecolors=color, lw=1.5)
    ax.set_xscale('log')
    ax.set(xlabel='Peak learning rate', ylabel='Best validation loss (nats/token)',
           title='Learning-rate search on the tuning seed (ring = selected)')
    ax.legend(loc='best')
    fig.savefig(path)
    plt.close(fig)


def plot_test_loss(runs, agg, path):
    """Mean test loss ± 1 sample std per method, with each seed as a dot."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    methods = [m for m in METHODS if m in agg and agg[m]['test_loss']['n']]
    for i, m in enumerate(methods):
        label, color, marker = SERIES[m]
        s = agg[m]['test_loss']
        seeds = [r['test']['test_loss'] for r in runs[m] if r['test']]
        ax.scatter([i + 0.15 + 0.05 * j for j in range(len(seeds))], seeds, color=color, alpha=0.45, s=18, marker=marker)
        std = 0 if math.isnan(s['std']) else s['std']
        ax.errorbar(i, s['mean'], yerr=std, color=color, marker=marker, ms=8, capsize=6, lw=2)
        ax.annotate(f"{s['mean']:.3f} ± {std:.3f}", (i, s['mean']), xytext=(-12, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=8.5)
    ax.set_xticks(range(len(methods)), [SERIES[m][0] for m in methods])
    ax.set_xlim(-0.9, len(methods) - 0.4)
    ax.grid(axis='x', visible=False)
    ax.set(ylabel='Test loss (nats/token)', title='Test loss at the selected checkpoint (mean ± std, dots = seeds)')
    fig.savefig(path)
    plt.close(fig)


def plot_memory(agg, path):
    """GPU memory components, mean over seeds, grouped by component and coloured by method."""
    plt = _plt()
    groups = [('model_after_load_mib', 'Model'), ('optimizer_state_mib', 'AdamW state'),
              ('peak_allocated_mib', 'Peak allocated'), ('peak_reserved_mib', 'Peak reserved')]
    methods = [m for m in METHODS if m in agg]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, m in enumerate(methods):
        label, color, _ = SERIES[m]
        ys = [agg[m][k]['mean'] for k, _ in groups]
        bars = ax.bar([g + (i - (len(methods) - 1) / 2) * width for g in range(len(groups))], ys, width * 0.92,
                      color=color, label=label)
        ax.bar_label(bars, labels=[f'{y:,.0f}' for y in ys], fontsize=7, color='#52514e', padding=2)
    ax.set_xticks(range(len(groups)), [g for _, g in groups])
    ax.grid(axis='x', visible=False)
    ax.set(ylabel='GPU memory (MiB)', title='GPU memory during training (mean over seeds)')
    ax.legend(loc='upper left')
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- markdown
def fmt(s, decimals):
    if not s['n']:
        return 'n/a'
    f = lambda v: f'{v:,.{decimals}f}'
    return f(s['mean']) + ('' if math.isnan(s['std']) else f" ± {f(s['std'])}")


def results_markdown(output, readme_dir, runs, failed, selected, agg):
    figures = output / 'figures'
    img = lambda name, alt: f'![{alt}]({os.path.relpath(figures / name, readme_dir)})'
    L = [START, '', f'_Generated by `python benchmark.py report` from `{os.path.relpath(output, readme_dir)}/`._', '']
    if not selected:
        return '\n'.join(L + ['**Pending.** The multi-seed benchmark has not been run yet '
                              '(`python benchmark.py all`). No numbers are shown until it has.', '', END])

    width = max(len(s['candidates']) for s in selected.values())
    L += ['**Learning-rate search** (tuning seed, best validation loss over the run):', '',
          '| Method | ' + ' | '.join(f'LR {i + 1}' for i in range(width)) + ' | Selected |',
          '|---|' + '---:|' * width + '---:|']
    edges = []
    for m, s in selected.items():
        cells = [f"{c['lr']:g}: " + (f"{c['best_validation_loss']:.4f}" if c['status'] == 'completed' else 'failed')
                 for c in s['candidates']]
        L.append(f"| {SERIES[m][0]} | {' | '.join(cells)} | {s['lr']:g} |")
        lrs = [c['lr'] for c in s['candidates']]
        if len(lrs) > 1 and s['lr'] in (min(lrs), max(lrs)):
            edges.append(SERIES[m][0])
    L.append('')
    if edges:
        L += [f"Selected LR is at the edge of the grid for: {', '.join(edges)}. The optimum may lie outside it.", '']
    L += [img('lr_search.png', 'Learning-rate search'), '']

    if not agg:
        L += ['**Final runs pending.**', '']
    else:
        present = [m for m in METHODS if m in agg]
        n = ', '.join(f"{SERIES[m][0]} {agg[m]['test_loss']['n']}" for m in present)
        L += [f'**Final runs** (mean ± sample std over seeds; n = {n}):', '',
              '| | ' + ' | '.join(SERIES[m][0] for m in present) + ' |', '|---|' + '---:|' * len(present)]
        L += [f'| {label} | ' + ' | '.join(fmt(agg[m][k], d) for m in present) + ' |'
              for k, (label, _, d) in METRICS.items()]
        L += ['', f'Per-seed numbers: `{os.path.relpath(output / "final_runs.csv", readme_dir)}`.', '']
        for name, alt in (('validation_vs_steps.png', 'Validation loss vs steps'),
                          ('validation_vs_time.png', 'Validation loss vs training time'),
                          ('test_loss.png', 'Test loss'), ('peak_memory.png', 'Peak GPU memory')):
            if (figures / name).exists():
                L += [img(name, alt), '']
    if failed:
        L += ['Failed or incomplete runs (excluded): ' + ', '.join(f'`{f}`' for f in failed), '']
    return '\n'.join(L + [END])


def update_readme(readme, block):
    """Replace the text between the results markers; refuse unless they appear exactly once, in order."""
    text = readme.read_text()
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise SystemExit(f'{readme}: expected exactly one {START} ... {END} block')
    readme.write_text(text[:text.index(START)] + block + text[text.index(END) + len(END):])


def write_results(output, readme=None):
    """Tables and figures under output/; the results block goes into readme if given, else output/results.md."""
    output = Path(output)
    runs, failed, selected = load(output)
    agg = aggregate(runs)
    figures = output / 'figures'
    if selected or agg:
        figures.mkdir(parents=True, exist_ok=True)
        write_tables(output, runs, agg, selected)
    if selected:
        plot_lr_search(selected, figures / 'lr_search.png')
    if agg:
        plot_curves(runs, figures / 'validation_vs_steps.png', 'step', 'Optimizer step', 'Validation loss vs steps')
        plot_curves(runs, figures / 'validation_vs_time.png', 'train_time_s',
                    'Training time (s, synchronized, evaluation excluded)', 'Validation loss vs training time')
        plot_test_loss(runs, agg, figures / 'test_loss.png')
        plot_memory(agg, figures / 'peak_memory.png')
    target = Path(readme) if readme else output / 'results.md'
    block = results_markdown(output, target.parent, runs, failed, selected, agg)
    if readme:
        update_readme(target, block)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(block + '\n')
    print(f'wrote results block to {target}')

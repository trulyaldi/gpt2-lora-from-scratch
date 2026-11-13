"""Full fine-tuning vs LoRA vs QLoRA (rank 8) of pretrained GPT-2 small on Tiny Shakespeare.

    python benchmark.py all                    # tune -> final -> test -> report, default preset
    python benchmark.py all --preset smoke     # a few minutes; writes results/smoke/
    python benchmark.py tune | final | test | report    # the same stages one at a time
    python benchmark.py run --config RUN_DIR/config.json  # one training run (used internally)

Stages:
  tune    one run per (method, learning rate) with --tune-seed; picks the LR with the lowest
          best-checkpoint validation loss per method -> selected_hparams.json. No test data is loaded.
  final   one run per (method, seed) with the selected LR, saving the lowest-validation-loss checkpoint.
  test    reloads each final checkpoint, re-checks its validation loss, then measures test loss.
  report  mean +- std over seeds, CSV/JSON tables and figures; rewrites the results block of README.md
          (only for the default results/ directory; other outputs get <output>/results.md).

Each training run is a separate process (fresh allocator, kernel caches and peak-memory counters).
Finished runs are skipped and interrupted ones resume from last.pt, so rerunning a stage continues it.
Failed runs (including CUDA out-of-memory) are recorded in failed.json, reported, and make the
command exit non-zero; they are never retried with different settings.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import HF_REPO, ROOT, load_splits, system_info

METHODS = ('full', 'lora', 'qlora')

PRESETS = {
    'default': dict(output='results', steps=1000, batch_size=8, micro_batch_size=4,
                    sequence_length=256, eval_every=50, warmup_steps=50, checkpoint_every=200, timing_skip=10,
                    seeds=[42, 123, 456], tune_seed=0,
                    lr_grid_full=[1e-5, 3e-5, 1e-4], lr_grid_adapter=[3e-4, 1e-3, 3e-3]),
    'smoke': dict(output='results/smoke', steps=20, batch_size=8,
                  micro_batch_size=4, sequence_length=256, eval_every=5, warmup_steps=5, checkpoint_every=10,
                  timing_skip=3, seeds=[42, 123], tune_seed=0,
                  lr_grid_full=[3e-5, 1e-4], lr_grid_adapter=[1e-3, 3e-3]),
}
FIXED = dict(min_lr_ratio=0.1, weight_decay=0.01, grad_clip=1.0, rank=8, alpha=16.0, dropout=0.0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('all', 'tune', 'final', 'test', 'report', 'run'))
    p.add_argument('--preset', choices=PRESETS, default='default')
    p.add_argument('--config', type=Path, help='run: config.json of a single run')
    p.add_argument('--output', type=Path, help='results directory (preset default)')
    p.add_argument('--readme', type=Path, help='README to update (default: README.md when --output is results/)')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    for name, kind in (('steps', int), ('batch_size', int), ('micro_batch_size', int), ('sequence_length', int),
                       ('eval_every', int), ('warmup_steps', int), ('checkpoint_every', int), ('timing_skip', int),
                       ('tune_seed', int), ('min_lr_ratio', float), ('weight_decay', float), ('grad_clip', float),
                       ('rank', int), ('alpha', float)):
        p.add_argument('--' + name.replace('_', '-'), type=kind)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--lr-grid-full', type=float, nargs='+')
    p.add_argument('--lr-grid-adapter', type=float, nargs='+', help='LoRA and QLoRA')
    p.add_argument('--no-deterministic', dest='deterministic', action='store_false',
                   help='disable torch.use_deterministic_algorithms (on by default)')
    args = p.parse_args(argv)
    for key, value in {**PRESETS[args.preset], **FIXED}.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.output = ROOT / args.output
    if args.stage == 'run' and args.config is None:
        p.error('run needs --config')
    if args.sequence_length > 1024:
        p.error('GPT-2 context is 1024 tokens')
    if args.batch_size % args.micro_batch_size:
        p.error('--batch-size must be a multiple of --micro-batch-size')
    if len({len(args.lr_grid_full), len(args.lr_grid_adapter)}) != 1:
        p.error('use the same number of learning-rate candidates for every method (equal tuning budget)')
    return args


def run_config(args, method, lr, seed, run_dir, save_best):
    """Everything one training run depends on; written to run_dir/config.json."""
    keys = ('steps', 'batch_size', 'micro_batch_size', 'sequence_length', 'eval_every', 'warmup_steps',
            'min_lr_ratio', 'weight_decay', 'grad_clip', 'rank', 'alpha', 'dropout', 'checkpoint_every',
            'timing_skip', 'deterministic')
    return {'method': method, 'lr': lr, 'seed': seed, **{k: getattr(args, k) for k in keys},
            'save_best': save_best, 'run_dir': os.path.relpath(run_dir, ROOT)}


def launch(cfg):
    """Run one training job in a subprocess. Returns True on success; on failure writes failed.json."""
    run_dir = ROOT / cfg['run_dir']
    if (run_dir / 'summary.json').exists():
        return True
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / 'config.json'
    if path.exists() and json.loads(path.read_text()) != cfg:
        raise SystemExit(f'{path} exists with a different configuration; use another --output or delete it')
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    (run_dir / 'failed.json').unlink(missing_ok=True)
    print(f"== {cfg['method']} lr={cfg['lr']:g} seed={cfg['seed']} -> {cfg['run_dir']}", flush=True)
    with open(run_dir / 'stderr.log', 'w') as err:
        code = subprocess.run([sys.executable, __file__, 'run', '--config', str(path)], cwd=ROOT, stderr=err).returncode
    if code == 0:
        return True
    tail = (run_dir / 'stderr.log').read_text().strip().splitlines()[-15:]
    (run_dir / 'failed.json').write_text(json.dumps({'returncode': code, 'stderr_tail': tail}, indent=2) + '\n')
    print(f'!! FAILED (exit {code}): {cfg["run_dir"]}\n' + '\n'.join(tail), file=sys.stderr, flush=True)
    return False


def lr_grid(args, method):
    return args.lr_grid_full if method == 'full' else args.lr_grid_adapter


def tune(args):
    """LR search on the tuning seed; selection by best validation loss only."""
    ok = True
    for method in args.methods:
        for lr in lr_grid(args, method):
            run_dir = args.output / 'runs' / 'tune' / method / f'lr{lr:g}'
            ok &= launch(run_config(args, method, lr, args.tune_seed, run_dir, save_best=False))
    selected = {}
    for method in args.methods:
        candidates = []
        for lr in lr_grid(args, method):
            run_dir = args.output / 'runs' / 'tune' / method / f'lr{lr:g}'
            if (run_dir / 'summary.json').exists():
                s = json.loads((run_dir / 'summary.json').read_text())
                candidates.append({'lr': lr, 'status': 'completed', 'best_validation_loss': s['best_validation_loss'],
                                   'best_step': s['best_step'], 'final_validation_loss': s['final_validation_loss']})
            else:
                candidates.append({'lr': lr, 'status': 'failed'})
        done = [c for c in candidates if c['status'] == 'completed']
        if not done:
            raise SystemExit(f'every tuning run failed for {method}; see failed.json under {args.output}/runs/tune')
        best = min(done, key=lambda c: c['best_validation_loss'])
        selected[method] = {'lr': best['lr'], 'selection': 'lowest best-checkpoint validation loss',
                            'tune_seed': args.tune_seed, 'candidates': candidates}
    (args.output / 'selected_hparams.json').write_text(json.dumps(selected, indent=2) + '\n')
    for method, s in selected.items():
        print(f"selected {method}: lr={s['lr']:g}")
    return ok


def final(args):
    path = args.output / 'selected_hparams.json'
    if not path.exists():
        raise SystemExit(f'{path} not found; run the tune stage first')
    selected = json.loads(path.read_text())
    ok = True
    for seed in args.seeds:  # seed-major: an interrupted sweep still has every method for the seeds done
        for method in args.methods:
            run_dir = args.output / 'runs' / 'final' / method / f'seed{seed}'
            ok &= launch(run_config(args, method, selected[method]['lr'], seed, run_dir, save_best=True))
    return ok


def test(args):
    """Test-set evaluation of every completed final run's best-validation checkpoint."""
    from training import evaluate_checkpoint
    ok = True
    for method in args.methods:
        for seed in args.seeds:
            run_dir = args.output / 'runs' / 'final' / method / f'seed{seed}'
            if not (run_dir / 'summary.json').exists():
                print(f'!! no completed run in {run_dir}; not tested', file=sys.stderr)
                ok = False
                continue
            r = evaluate_checkpoint(run_dir)
            print(f"{method} seed {seed}: checkpoint step {r['checkpoint_step']}, test loss {r['test_loss']:.4f}")
    return ok


def hardware():
    """system_info() plus PyTorch, CUDA, GPU and NVIDIA driver details."""
    import torch
    info = system_info()
    info.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
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


def write_environment(args):
    """Record config, split metadata and hardware once per output directory (updated each invocation)."""
    _, split_meta = load_splits(())
    config = {k: (os.path.relpath(v, ROOT) if isinstance(v, Path) else v) for k, v in vars(args).items()
              if k not in ('stage', 'config', 'readme')}
    env = {'config': config, 'data': split_meta,
           'model': {'name': 'GPT-2 small (124M)', 'weights': f'huggingface.co/{HF_REPO} model.safetensors'},
           'reproducibility': {
               'seeding': 'Python random, NumPy, torch CPU and CUDA seeded per run; data order from a '
                          'torch.Generator seeded with the run seed',
               'deterministic_algorithms': args.deterministic, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'note': 'Deterministic mode makes reruns on the same machine and software repeatable in '
                       'practice; it is not a guarantee of bitwise-identical results across GPUs, drivers '
                       'or library versions.'},
           'hardware': hardware(), 'command': ' '.join(sys.argv)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'environment.json').write_text(json.dumps(env, indent=2) + '\n')


def main():
    args = parse_args()
    if args.stage == 'run':
        from training import train
        train(json.loads(args.config.read_text()))
        return
    if args.stage != 'report':
        write_environment(args)
    ok = True
    if args.stage in ('all', 'tune'):
        ok &= tune(args)
    if args.stage in ('all', 'final'):
        ok &= final(args)
    if args.stage in ('all', 'test'):
        ok &= test(args)
    if args.stage in ('all', 'report'):
        from analysis import write_results
        readme = args.readme or (ROOT / 'README.md' if args.output == ROOT / 'results' else None)
        write_results(args.output, readme)
    if not ok:
        sys.exit('one or more runs failed or are missing; see the messages above and failed.json files')


if __name__ == '__main__':
    main()

"""Training correctness: leakage-free splits, batch order, which weights each method updates, gradient
accumulation, evaluation, checkpoints, and a short CUDA run of every method."""
import copy
import json
import math
import statistics

import pytest

torch = pytest.importorskip('torch')
import common
import training
from gpt_torch import GPT, GPTConfig

TINY = GPTConfig(vocab_size=97, block_size=32, n_layer=2, n_head=2, n_embd=16)


def test_split_text_is_contiguous_disjoint_and_cut_at_blank_lines():
    text = ''.join(f'SPEAKER {i}:\nline {i} of the play.\n\n' for i in range(500))
    bounds = common.split_text(text)
    assert list(bounds) == ['train', 'validation', 'test']
    (a0, a1), (b0, b1), (c0, c1) = bounds.values()
    assert a0 == 0 and a1 == b0 and b1 == c0 and c1 == len(text)
    assert text[a0:a1] + text[b0:b1] + text[c0:c1] == text
    for cut in (a1, b1):
        assert text[cut - 2:cut] == '\n\n'
    for (s, e), frac in zip(bounds.values(), (0.8, 0.1, 0.1)):
        assert abs((e - s) / len(text) - frac) < 0.01


def test_tiny_shakespeare_splits_are_tokenized_separately_and_test_is_opt_in():
    pytest.importorskip('tiktoken')
    import tiktoken
    tokens, meta = common.load_splits()
    text = common.read_shakespeare()
    enc = tiktoken.get_encoding('gpt2')
    for name, s in meta['splits'].items():
        assert tokens[name].tolist() == enc.encode_ordinary(text[s['char_start']:s['char_end']])
        assert s['tokens'] == len(tokens[name])
    assert meta['splits']['train']['char_end'] == meta['splits']['validation']['char_start']
    assert meta['splits']['validation']['char_end'] == meta['splits']['test']['char_start']
    assert set(common.load_splits(('train', 'validation'))[0]) == {'train', 'validation'}


def test_batch_positions_depend_only_on_seed_and_stay_inside_the_split():
    a = training.batch_positions(42, 50, 8, 1000, 64)
    assert torch.equal(a, training.batch_positions(42, 50, 8, 1000, 64))
    assert not torch.equal(a, training.batch_positions(123, 50, 8, 1000, 64))
    assert a.shape == (50, 8) and a.min() >= 0 and a.max() + 64 + 1 <= 1000  # input + shifted target fit


def tiny_model(method, seed=0):
    torch.manual_seed(seed)
    return training.wrap(GPT(TINY).double(), method, rank=4, alpha=8)


def train_steps(model, n, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    data = torch.randint(TINY.vocab_size, (400,))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    grads = []
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        training.accumulate_gradients(model, data, torch.randint(380, (4,)), 2, 16, 'cpu')
        grads.append({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
        opt.step()
    return opt, grads


def test_full_finetuning_updates_pretrained_weights():
    model = tiny_model('full')
    before = copy.deepcopy(model.state_dict())
    _, grads = train_steps(model, 1)
    assert all(p.requires_grad for p in model.parameters())
    for name in ('wte.weight', 'h.0.attn.c_attn.weight', 'h.1.mlp.c_proj.weight', 'ln_f.weight'):
        assert grads[0][name].abs().sum() > 0
        assert not torch.equal(before[name], model.state_dict()[name])


@pytest.mark.parametrize('method', ['lora', 'qlora'])
def test_adapters_train_and_base_stays_frozen(method):
    model = tiny_model(method)
    frozen_before = {k: v.clone() for k, v in model.state_dict().items() if 'lora_' not in k}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(('lora_A', 'lora_B')) for n in trainable)
    assert len(trainable) == 2 * 4 * TINY.n_layer  # A and B on four projections per block
    opt, grads = train_steps(model, 2)
    assert set(grads[0]) == trainable  # no gradient reaches any frozen tensor
    for n in trainable:  # B starts at zero, so A's gradient is exactly zero on step 1 and nonzero after B moves
        assert grads[0][n].abs().sum() > 0 if n.endswith('lora_B') else grads[0][n].abs().sum() == 0
        assert grads[1][n].abs().sum() > 0
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in frozen_before.items())
    if method == 'qlora':
        assert any(k.endswith('base.packed') for k in frozen_before) and any(k.endswith('base.absmax') for k in frozen_before)
    state = sum(t.numel() for s in opt.state.values() for t in s.values() if t.dim() > 0)
    assert state == 2 * sum(model.get_parameter(n).numel() for n in trainable)  # AdamW holds only adapters


def test_gradient_accumulation_matches_one_full_batch():
    model = tiny_model('full')
    data = torch.randint(TINY.vocab_size, (300,), generator=torch.Generator().manual_seed(1))
    rows = torch.tensor([3, 50, 120, 200, 7, 260])
    loss = training.accumulate_gradients(model, data, rows, 2, 16, 'cpu')
    acc = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    x, y = training.windows(data, rows, 16, 'cpu')
    full = model(x, y)[1]
    full.backward()
    assert loss.item() == pytest.approx(full.item(), rel=1e-12)
    for n, p in model.named_parameters():
        torch.testing.assert_close(acc[n], p.grad, rtol=1e-10, atol=1e-12)


def test_evaluate_is_token_weighted_mean_over_all_non_overlapping_windows():
    model = tiny_model('full').eval()
    data = torch.randint(TINY.vocab_size, (16 * 5 + 3,), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        manual = statistics.fmean(model(data[None, s:s + 16], data[None, s + 1:s + 17])[1].item()
                                  for s in range(0, 16 * 5, 16))
    assert training.evaluate(model, data, 16, 2, 'cpu') == pytest.approx(manual, rel=1e-12)


@pytest.mark.parametrize('method', ['full', 'lora', 'qlora'])
def test_trainable_checkpoint_roundtrip(method):
    model = tiny_model(method)
    train_steps(model, 2)
    fresh = tiny_model(method)
    training.load_trainable_state(fresh, training.trainable_state(model))
    x = torch.randint(TINY.vocab_size, (2, 16))
    torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)
    with pytest.raises(ValueError):
        training.load_trainable_state(fresh, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('method', ['full', 'lora', 'qlora'])
def test_cuda_training_run_never_loads_test_and_checkpoint_reloads(method, tmp_path, monkeypatch):
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(common.HF_REPO, 'model.safetensors', local_files_only=True)
    except OSError as exc:
        pytest.skip(f'GPT-2 weights not in local cache: {exc}')
    requested = []
    real = common.load_splits
    monkeypatch.setattr(training, 'load_splits', lambda names, **kw: (requested.append(tuple(names)), real(names, **kw))[1])
    cfg = dict(method=method, lr=1e-3 if method != 'full' else 1e-4, seed=0, steps=4, batch_size=4, micro_batch_size=2,
               sequence_length=64, eval_every=2, warmup_steps=2, min_lr_ratio=0.1, weight_decay=0.01, grad_clip=1.0,
               rank=8, alpha=16.0, dropout=0.0, checkpoint_every=2, timing_skip=1, deterministic=True, save_best=True,
               run_dir=str(tmp_path))
    s = training.train(cfg)
    assert requested == [('train', 'validation')]
    assert set(s['data']) == {'train', 'validation'}
    assert 'test' not in json.dumps({k: v for k, v in s.items() if k != 'config'})  # config holds the tmp path
    assert s['memory_mib']['optimizer_state'] == pytest.approx(2 * s['memory_mib']['trainable_parameters'])
    assert s['best_validation_loss'] < s['initial_validation_loss']
    assert not (tmp_path / 'last.pt').exists() and (tmp_path / 'best.pt').exists()
    r = training.evaluate_checkpoint(tmp_path)
    assert r['checkpoint_step'] == s['best_step'] and math.isfinite(r['test_loss'])

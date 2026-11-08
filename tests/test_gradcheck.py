"""Finite-difference checks of every handwritten NumPy backward pass, plus causality,
GPT-2 initial loss, and an overfit-one-batch test of the NumPy AdamW."""
import numpy as np
import pytest

import gpt_numpy as g
from gpt_numpy import numerical_grad, rel_error

FULL_CHECK_TOL = 1e-7   # every entry checked, float64; measured errors are <= 2.1e-9
SPOT_CHECK_TOL = 1e-6   # sampled entries at GPT-2 small width; measured errors are <= 8.6e-8


def layer_errors(layer, x, rng):
    """Relative errors of dx and every parameter gradient for loss = sum(layer(x) * G)."""
    G = rng.normal(size=layer.forward(x).shape)
    f = lambda: np.sum(layer.forward(x) * G)
    f()
    dx = layer.backward(G)
    errors = {'dx': rel_error(dx, numerical_grad(f, x))}
    for name, p, grad in (layer.parameters() if hasattr(layer, 'parameters') else []):
        errors[name] = rel_error(grad, numerical_grad(f, p))
    return errors


def randomize(layer, rng, scale):
    """Move parameters away from their initial values so gains/biases are really exercised."""
    for _, p, _ in layer.parameters():
        p[...] = p + rng.normal(size=p.shape) * scale


def make_linear(rng):
    return g.Linear(4, 5, std=1.0, rng=rng)


def make_mlp(rng):
    return g.MLP(4, n_layer=1, rng=rng)


def make_layernorm(rng):
    return g.LayerNorm(6)


def make_attention(rng):
    return g.CausalSelfAttention(6, n_head=2, n_layer=1, rng=rng)


def make_block(rng):
    return g.Block(6, n_head=2, n_layer=1, rng=rng)


@pytest.mark.parametrize('make, shape, scale', [
    (make_linear, (2, 3, 4), 1.0),
    (lambda rng: g.GELU(), (2, 3, 4), None),
    (make_mlp, (2, 3, 4), 1.0),
    (lambda rng: g.Softmax(), (2, 2, 3, 3), None),
    (make_layernorm, (2, 3, 6), 1.0),
    (make_attention, (2, 4, 6), 0.7),
    (make_block, (2, 4, 6), 0.5),
], ids=['linear', 'gelu', 'mlp', 'softmax', 'layernorm', 'attention', 'block'])
def test_layer_gradients(make, shape, scale):
    rng = np.random.default_rng(0)
    layer = make(rng)
    if scale is not None:
        randomize(layer, rng, scale)
    errors = layer_errors(layer, rng.normal(size=shape) * 2, rng)
    assert max(errors.values()) < FULL_CHECK_TOL, errors


def test_embedding_gradient_with_repeated_tokens():
    rng = np.random.default_rng(0)
    emb = g.Embedding(5, 3, std=1.0, rng=rng)
    idx = np.array([[1, 2, 1, 1, 0, 4]])        # token 1 repeats: its gradient rows must add up
    G = rng.normal(size=(1, 6, 3))
    f = lambda: np.sum(emb.forward(idx) * G)
    f()
    emb.backward(G)
    assert rel_error(emb.grads['W'], numerical_grad(f, emb.params['W'])) < FULL_CHECK_TOL


def test_cross_entropy_gradient():
    rng = np.random.default_rng(0)
    ce = g.CrossEntropyLoss()
    logits, targets = rng.normal(size=(2, 3, 5)), rng.integers(0, 5, size=(2, 3))
    f = lambda: ce.forward(logits, targets)
    f()
    assert rel_error(ce.backward(), numerical_grad(f, logits)) < FULL_CHECK_TOL


def test_full_tiny_gpt_gradients():
    """Every gradient of a 2-layer GPT, including the tied embedding/output head."""
    rng = np.random.default_rng(0)
    model = g.GPT(vocab_size=7, block_size=5, n_layer=2, n_head=2, n_embd=8, seed=3)
    for name, p, _ in model.parameters():
        if name.endswith('W'):
            p *= 10                              # larger weights give a less trivial check
    idx, targets = rng.integers(0, 7, size=(2, 4)), rng.integers(0, 7, size=(2, 4))
    ce = g.CrossEntropyLoss()
    f = lambda: ce.forward(model.forward(idx), targets)
    model.zero_grad()
    f()
    model.backward(ce.backward())
    errors = {name: rel_error(grad, numerical_grad(f, p)) for name, p, grad in model.parameters()}
    assert max(errors.values()) < FULL_CHECK_TOL, errors


def spot_check(f, p, grad, k=5, eps=1e-5, seed=123):
    """Worst relative error over k random entries of grad with non-negligible magnitude."""
    r = np.random.default_rng(seed)
    candidates = np.flatnonzero(np.abs(grad) > 1e-3 * np.abs(grad).max())
    worst = 0.0
    for i in r.choice(candidates, size=min(k, candidates.size), replace=False):
        ii = np.unravel_index(i, p.shape)
        old = p[ii]
        p[ii] = old + eps; fp = f()
        p[ii] = old - eps; fm = f()
        p[ii] = old
        num = (fp - fm) / (2 * eps)
        worst = max(worst, abs(num - grad[ii]) / (abs(num) + abs(grad[ii])))
    return worst


def test_full_width_block_spot_check():
    """One GPT-2 small block (C=768, 12 heads): sampled entries of dx and every parameter gradient."""
    rng = np.random.default_rng(0)
    block = g.Block(768, n_head=12, n_layer=12, rng=rng)
    x, G = rng.normal(size=(1, 8, 768)), rng.normal(size=(1, 8, 768))
    f = lambda: np.sum(block.forward(x) * G)
    f()
    dx = block.backward(G)
    worst = max([spot_check(f, x, dx)] + [spot_check(f, p, grad) for _, p, grad in block.parameters()])
    assert worst < SPOT_CHECK_TOL


def test_attention_is_causal():
    rng = np.random.default_rng(0)
    attn = g.CausalSelfAttention(16, n_head=4, n_layer=1, rng=rng)
    x = rng.normal(size=(2, 8, 16))
    out = attn.forward(x).copy()
    x2 = x.copy()
    x2[:, 5] += 5.0                              # change position 5 only
    changed = np.abs(attn.forward(x2) - out).max(axis=(0, 2)) > 1e-12
    assert changed.tolist() == [False] * 5 + [True] * 3
    attn.forward(x)
    G = np.zeros_like(x)
    G[:, 2] = 1.0                                # gradient on output position 2 only
    reached = np.abs(attn.backward(G)).max(axis=(0, 2)) > 1e-12
    assert reached.tolist() == [True] * 3 + [False] * 5


def test_gpt2_small_initial_loss():
    """At GPT-2 small init the loss should be close to ln(V) + C * 0.02**2 / 2 ~= 10.98."""
    rng = np.random.default_rng(0)
    model = g.GPT(dtype=np.float32)
    x, y = rng.integers(0, 50257, size=(1, 64)), rng.integers(0, 50257, size=(1, 64))
    loss = g.CrossEntropyLoss().forward(model.forward(x), y)
    assert abs(loss - (np.log(50257) + 768 * 0.02**2 / 2)) < 0.25


def test_adamw_overfits_one_batch():
    rng = np.random.default_rng(0)
    model = g.GPT(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16, seed=0)
    params = model.parameters()
    optimizer = g.AdamW(params, lr=1e-2, weight_decay=0.0)
    ce = g.CrossEntropyLoss()
    x, y = rng.integers(0, 11, size=(2, 8)), rng.integers(0, 11, size=(2, 8))
    first = None
    for _ in range(100):
        model.zero_grad()
        loss = ce.forward(model.forward(x), y)
        first = loss if first is None else first
        model.backward(ce.backward())
        g.clip_grad_norm(params, 1.0)
        optimizer.step()
    assert first > 2.0 and loss < 0.05, (first, loss)

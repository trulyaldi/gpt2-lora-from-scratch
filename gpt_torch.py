"""PyTorch port of gpt_numpy.GPT. Linear weights use PyTorch's (out, in) layout."""
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.n_embd
        assert c % config.n_head == 0
        self.n_head = config.n_head
        self.c_attn = nn.Linear(c, 3 * c)
        self.c_proj = nn.Linear(c, c)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=-1)
        def heads(z):
            return z.reshape(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(c // self.n_head))
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        probs = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1)
        y = (probs @ v).transpose(1, 2).contiguous().reshape(b, t, c)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config=GPTConfig()):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.apply(self._init_weights)
        for block in self.h:
            for layer in (block.attn.c_proj, block.mlp.c_proj):
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or idx.shape[1] > self.config.block_size:
            raise ValueError('input must be (batch, time) within block_size')
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        logits = F.linear(self.ln_f(x), self.wte.weight)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=50):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        for _ in range(max_new):
            logits = self(idx[:, -self.config.block_size:])[:, -1, :] / temperature
            if top_k:
                cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, -torch.inf)
            idx = torch.cat((idx, torch.multinomial(logits.softmax(-1), 1)), dim=1)
        return idx


def copy_from_numpy(torch_model, numpy_model):
    """Copy named NumPy parameters; transpose only Linear weights."""
    return copy_from_arrays(torch_model, {name: p for name, p, _ in numpy_model.parameters()})


def load_pretrained_gpt2(path, dtype=torch.float32):
    """Build GPT-2 small from a HuggingFace model.safetensors file, without a NumPy model copy."""
    from gpt_numpy import load_safetensors, _hf_name_to_ours
    src = {}
    for key, array in load_safetensors(path).items():
        name = _hf_name_to_ours(key)
        if name is not None:
            src[name] = array
    model = GPT(GPTConfig()).to(dtype)
    return copy_from_arrays(model, src)


def copy_from_arrays(torch_model, src):
    """Copy arrays named like gpt_numpy parameters ('h.0.attn.c_attn.W', '.b', '.g');
    NumPy/HF weights are (in, out), so Linear weights are transposed. Every array must be used."""
    used = set()
    def take(key):
        used.add(key)
        return torch.from_numpy(np.array(src[key]))  # copy: safetensors arrays are read-only memmaps
    with torch.no_grad():
        for name, module in torch_model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(take(name + '.W').T)
                module.bias.copy_(take(name + '.b'))
            elif isinstance(module, nn.Embedding):
                module.weight.copy_(take(name + '.W'))
            elif isinstance(module, nn.LayerNorm):
                module.weight.copy_(take(name + '.g'))
                module.bias.copy_(take(name + '.b'))
    if used != set(src):
        raise ValueError(f'unused source arrays: {sorted(set(src) - used)[:5]}')
    return torch_model

"""Minimal LoRA for GPT's nn.Linear projections."""
import math
import torch
from torch import nn
from torch.nn import functional as F

TARGETS = ('attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('expected nn.Linear, rank/alpha > 0, and 0 <= dropout < 1')
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling

    @torch.no_grad()
    def merge(self):
        self.base.weight.add_((self.lora_B @ self.lora_A) * self.scaling)
        return self.base


def apply_lora(model, rank=8, alpha=16, dropout=0.0, targets=TARGETS):
    model.requires_grad_(False)
    matched = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(s) for s in targets):
            parent_name, attr = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, attr, LoRALinear(module, rank, alpha, dropout))
            matched.append(name)
    if not matched:
        raise ValueError('no target linear modules found')
    return matched


def merge_lora(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), attr, module.merge())
    return model


def adapter_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name.endswith(('.lora_A', '.lora_B'))}


def save_adapter(model, path):
    modules = {name: {'rank': m.rank, 'alpha': m.alpha, 'dropout': m.dropout.p}
               for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    torch.save({'modules': modules, 'weights': adapter_state_dict(model)}, path)


def load_adapter(model, path):
    package = torch.load(path, map_location='cpu', weights_only=True)
    for name, cfg in package['modules'].items():
        module = model.get_submodule(name)
        if not isinstance(module, LoRALinear) or (module.rank, module.alpha, module.dropout.p) != (cfg['rank'], cfg['alpha'], cfg['dropout']):
            raise ValueError(f'adapter configuration mismatch: {name}')
    expected = set(adapter_state_dict(model))
    if set(package['weights']) != expected:
        raise ValueError('adapter target mismatch')
    model.load_state_dict(package['weights'], strict=False)
    return model

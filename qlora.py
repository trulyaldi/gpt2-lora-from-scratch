"""Minimal QLoRA: 4-bit NormalFloat (NF4) frozen base weights plus a trainable LoRA update.

The base weight of each target projection is stored as blockwise-absmax NF4 codes, two per byte,
with one fp32 scale per block of 64 values (bitsandbytes' default layout). The forward pass
dequantizes it on the fly; the backward pass dequantizes it again instead of keeping the
full-precision copy alive, which is where the memory saving comes from. Compute stays in the
dtype of the LoRA matrices (fp32 here). Double quantization of the scales and paged optimizers
from the QLoRA paper are not implemented: at GPT-2 small scale they would save only a few MiB.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from lora import TARGETS, LoRALinear

# The 16 NF4 levels: quantiles of N(0, 1) scaled to [-1, 1], with an exact zero
# (QLoRA paper, Dettmers et al. 2023; values as in bitsandbytes.functional.get_4bit_type('nf4')).
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
       0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)
BLOCK_SIZE = 64


def quantize_nf4(weight, block_size=BLOCK_SIZE):
    """Return (packed uint8 codes, fp32 absmax per block) for a float tensor.
    The tensor is flattened row-major into blocks of block_size values (block_size must be even
    and divide the size). Each value maps to the nearest NF4 level after scaling by 1/absmax of its
    block; the first value of each pair goes in the high nibble (same layout as bitsandbytes)."""
    flat = weight.detach().float().reshape(-1)
    if block_size % 2 or flat.numel() % block_size:
        raise ValueError(f'block_size={block_size} must be even and divide the number of elements')
    blocks = flat.reshape(-1, block_size)
    absmax = blocks.abs().amax(dim=1)
    scaled = blocks * (1.0 / absmax.clamp_min(torch.finfo(torch.float32).tiny))[:, None]  # as bitsandbytes does
    levels = torch.tensor(NF4, dtype=torch.float64)
    midpoints = ((levels[1:] + levels[:-1]) / 2).to(flat.device, torch.float32)
    codes = torch.bucketize(scaled.reshape(-1), midpoints).to(torch.uint8)  # nearest level
    packed = (codes[0::2] << 4) | codes[1::2]
    return packed, absmax


def dequantize_nf4(packed, absmax, shape, dtype=torch.float32, block_size=BLOCK_SIZE):
    """Inverse of quantize_nf4: NF4 level of each code times its block's absmax, reshaped."""
    codes = torch.stack((packed >> 4, packed & 0x0F), dim=1).reshape(-1).long()
    levels = torch.tensor(NF4, device=packed.device, dtype=torch.float32)
    values = levels[codes].reshape(-1, block_size) * absmax[:, None]
    return values.reshape(shape).to(dtype)


class _NF4Linear(torch.autograd.Function):
    """y = x @ dequant(W).T + b with W frozen. Only the packed codes are kept for backward."""
    @staticmethod
    def forward(ctx, x, packed, absmax, bias, shape):
        weight = dequantize_nf4(packed, absmax, shape, x.dtype)
        ctx.save_for_backward(packed, absmax)
        ctx.shape = shape
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        packed, absmax = ctx.saved_tensors
        weight = dequantize_nf4(packed, absmax, ctx.shape, grad_out.dtype)
        return grad_out @ weight, None, None, None, None  # base weight and bias are frozen


class NF4Linear(nn.Module):
    """Frozen linear layer whose weight is stored in NF4; the bias stays full precision."""
    def __init__(self, linear, block_size=BLOCK_SIZE):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise ValueError('expected nn.Linear')
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.block_size = block_size
        packed, absmax = quantize_nf4(linear.weight, block_size)
        self.register_buffer('packed', packed)
        self.register_buffer('absmax', absmax)
        bias = None if linear.bias is None else linear.bias.detach().clone()
        self.register_buffer('bias', bias)

    @property
    def shape(self):
        return (self.out_features, self.in_features)

    def dequantized_weight(self, dtype=torch.float32):
        return dequantize_nf4(self.packed, self.absmax, self.shape, dtype, self.block_size)

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return _NF4Linear.apply(x, self.packed, self.absmax, bias, self.shape)


class QLoRALinear(LoRALinear):
    """LoRALinear whose frozen base is an NF4Linear; A and B are full precision (fp32 by default)."""
    def __init__(self, base, rank, alpha, dropout=0.0, dtype=torch.float32):
        if not isinstance(base, NF4Linear) or rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('expected NF4Linear, rank/alpha > 0, and 0 <= dropout < 1')
        nn.Module.__init__(self)
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        device = base.packed.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def merge(self):
        raise NotImplementedError('merging into a 4-bit base would re-quantize it; '
                                  'dequantize the base and merge into an nn.Linear instead')


def apply_qlora(model, rank=8, alpha=16, dropout=0.0, targets=TARGETS, block_size=BLOCK_SIZE):
    """Freeze the model, quantize every target nn.Linear to NF4 and wrap it with LoRA.
    The original full-precision weights of those layers are released. Returns the wrapped names."""
    model.requires_grad_(False)
    matched = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(s) for s in targets):
            parent_name, attr = name.rsplit('.', 1)
            dtype = module.weight.dtype
            layer = QLoRALinear(NF4Linear(module, block_size), rank, alpha, dropout, dtype)
            setattr(model.get_submodule(parent_name), attr, layer)
            matched.append(name)
    if not matched:
        raise ValueError('no target linear modules found')
    return matched

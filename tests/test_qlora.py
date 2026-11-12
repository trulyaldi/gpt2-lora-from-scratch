"""QLoRA: NF4 quantization against bitsandbytes, the 4-bit layer against LoRA on the
dequantized weight, the custom backward against finite differences, and memory layout."""
import copy
import pytest

torch = pytest.importorskip('torch')
from gpt_torch import GPT, GPTConfig
from lora import LoRALinear, adapter_state_dict, save_adapter, load_adapter
from qlora import NF4, NF4Linear, QLoRALinear, _NF4Linear, apply_qlora, dequantize_nf4, quantize_nf4

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='bitsandbytes 4-bit kernels need CUDA')


def unpack(packed):
    return torch.stack((packed >> 4, packed & 0x0F), dim=1).reshape(-1)


def test_nf4_roundtrip_and_packing():
    torch.manual_seed(0)
    w = torch.randn(48, 32)
    packed, absmax = quantize_nf4(w)
    assert packed.dtype == torch.uint8 and packed.numel() == w.numel() // 2
    assert absmax.shape == (w.numel() // 64,)
    deq = dequantize_nf4(packed, absmax, w.shape)
    # every value lands on the nearest NF4 level of its block
    levels = torch.tensor(NF4)
    scaled = (w.reshape(-1, 64) / absmax[:, None]).reshape(-1, 1)
    nearest = (scaled - levels).abs().argmin(dim=1)
    torch.testing.assert_close(unpack(packed).long(), nearest, rtol=0, atol=0)
    # the block maximum is exact and the error is at most half the widest gap between levels
    blocks = w.reshape(-1, 64)
    torch.testing.assert_close(deq.reshape(-1, 64).abs().amax(1), blocks.abs().amax(1), rtol=0, atol=0)
    gap = max(b - a for a, b in zip(NF4, NF4[1:]))
    assert ((deq - w).abs().reshape(-1, 64) <= absmax[:, None] * gap / 2 + 1e-7).all()
    # re-quantizing a dequantized tensor is lossless
    p2, a2 = quantize_nf4(deq)
    assert p2.equal(packed) and a2.equal(absmax)


@cuda
def test_nf4_matches_bitsandbytes():
    bnb = pytest.importorskip('bitsandbytes.functional')
    assert torch.tensor(NF4).equal(bnb.get_4bit_type('nf4', device='cpu'))
    torch.manual_seed(1)
    mismatched, total = 0, 0
    for shape in [(2304, 768), (768, 3072), (3072, 768), (64, 128)]:
        w = torch.randn(shape, device='cuda') * 0.02
        packed, absmax = quantize_nf4(w)
        ref, state = bnb.quantize_4bit(w, blocksize=64, quant_type='nf4', compress_statistics=False)
        assert absmax.equal(state.absmax)
        ours, theirs = unpack(packed).long(), unpack(ref.reshape(-1)).long()
        diff = ours != theirs
        mismatched += diff.sum().item(); total += diff.numel()
        # bitsandbytes' CUDA kernel rounds the scaled value slightly differently, so a value lying
        # within float32 rounding of the midpoint between two levels can go either way
        assert ((ours - theirs).abs() <= 1).all()
        if diff.any():
            scaled = (w.reshape(-1, 64) * (1.0 / absmax)[:, None]).reshape(-1)[diff]
            lo = torch.minimum(ours, theirs)[diff].cpu()
            midpoint = (torch.tensor(NF4, dtype=torch.float64)[lo] + torch.tensor(NF4, dtype=torch.float64)[lo + 1]) / 2
            assert ((scaled.double().cpu() - midpoint).abs() < 1e-6).all()
        same = ~diff.reshape(shape)
        ours_w, theirs_w = dequantize_nf4(packed, absmax, shape), bnb.dequantize_4bit(ref, state)
        torch.testing.assert_close(ours_w[same], theirs_w[same], rtol=0, atol=0)
    assert mismatched / total < 1e-6


def test_qlora_matches_lora_on_dequantized_weights(tmp_path):
    torch.manual_seed(2)
    cfg = GPTConfig(29, 16, 2, 2, 32)
    model = GPT(cfg).double()
    targets = apply_qlora(model, rank=3, alpha=6)
    assert len(targets) == 8
    assert all(p.requires_grad == ('lora_' in n) for n, p in model.named_parameters())
    assert not any(isinstance(m, torch.nn.Linear) for m in model.modules())  # full-precision weights released
    # reference: plain LoRA whose base nn.Linear holds the dequantized weights
    reference = copy.deepcopy(model)
    for name, m in list(reference.named_modules()):
        if isinstance(m, QLoRALinear):
            linear = torch.nn.Linear(m.base.in_features, m.base.out_features).double()
            with torch.no_grad():
                linear.weight.copy_(m.base.dequantized_weight(torch.float64))
                linear.bias.copy_(m.base.bias)
            plain = LoRALinear(linear, m.rank, m.alpha)
            with torch.no_grad():
                plain.lora_A.copy_(m.lora_A)
            parent, attr = name.rsplit('.', 1)
            setattr(reference.get_submodule(parent), attr, plain)
    for module in (model, reference):
        for m in module.modules():
            if isinstance(m, LoRALinear):
                torch.manual_seed(5)
                m.lora_B.data.normal_(std=0.05)
    idx = torch.randint(0, 29, (2, 7))
    y = torch.randint(0, 29, (2, 7))
    out, loss = model(idx, y)
    out_ref, loss_ref = reference(idx, y)
    torch.testing.assert_close(out, out_ref, rtol=1e-12, atol=1e-12)
    loss.backward(); loss_ref.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    grads_ref = {n: p.grad for n, p in reference.named_parameters() if p.requires_grad}
    assert grads.keys() == grads_ref.keys() and len(grads) == 16
    for n in grads:
        torch.testing.assert_close(grads[n], grads_ref[n], rtol=1e-10, atol=1e-12)
    # adapters save and load through the same functions as plain LoRA
    path = tmp_path / 'adapter.pt'
    save_adapter(model, path)
    other = copy.deepcopy(model)
    for m in other.modules():
        if isinstance(m, QLoRALinear):
            m.lora_B.data.zero_()
    load_adapter(other, path)
    torch.testing.assert_close(other(idx), out.detach(), rtol=0, atol=0)
    assert set(adapter_state_dict(other)) == set(adapter_state_dict(model))


def test_nf4_linear_backward_gradcheck():
    torch.manual_seed(3)
    layer = NF4Linear(torch.nn.Linear(64, 6).double())
    x = torch.randn(3, 64, dtype=torch.float64, requires_grad=True)
    bias = layer.bias
    assert torch.autograd.gradcheck(lambda t: _NF4Linear.apply(t, layer.packed, layer.absmax, bias, layer.shape), (x,))


def test_nf4_linear_storage_and_saved_tensors():
    layer = NF4Linear(torch.nn.Linear(768, 2304))
    assert layer.packed.numel() == 768 * 2304 // 2 and layer.absmax.numel() == 768 * 2304 // 64
    assert [n for n, _ in layer.named_parameters()] == []
    saved = []
    x = torch.randn(4, 768, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        layer(x).sum().backward()
    # backward keeps only the packed codes and scales, never a dequantized fp32 weight
    assert all(t.numel() < 768 * 2304 for t in saved)
    assert {t.dtype for t in saved} <= {torch.uint8, torch.float32}
    with pytest.raises(ValueError):
        quantize_nf4(torch.zeros(100))

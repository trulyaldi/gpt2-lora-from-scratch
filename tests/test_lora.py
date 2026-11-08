"""Custom LoRA: freezing, zero-init output, gradients, adapter save/load, and merging."""
import copy
import pytest

torch = pytest.importorskip('torch')
from gpt_torch import GPT, GPTConfig
from lora import LoRALinear, apply_lora, merge_lora, save_adapter, load_adapter, adapter_state_dict


def test_lora_freeze_initial_grad_merge_save(tmp_path):
    torch.manual_seed(4)
    cfg = GPTConfig(23, 12, 2, 2, 8)
    model = GPT(cfg).double()
    idx = torch.randint(0, 23, (2, 6))
    initial = model(idx).detach()
    base = copy.deepcopy(model)
    targets = apply_lora(model, rank=3, alpha=6, dropout=0)
    assert len(targets) == 8
    torch.testing.assert_close(model(idx), initial, rtol=0, atol=0)
    assert all(p.requires_grad == ('lora_' in n) for n, p in model.named_parameters())
    loss = model(idx, idx)[1]
    loss.backward()
    assert all(p.grad is None for n, p in model.named_parameters() if 'lora_' not in n)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if 'lora_B' in n)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_B.data.normal_(std=0.01)
    changed = model(idx).detach()
    path = tmp_path / 'adapter.pt'
    save_adapter(model, path)
    apply_lora(base, rank=3, alpha=6)
    load_adapter(base, path)
    torch.testing.assert_close(base(idx), changed)
    assert set(adapter_state_dict(base)) == set(adapter_state_dict(model))
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    torch.testing.assert_close(model(idx), changed, rtol=1e-12, atol=1e-12)


def test_lora_shapes():
    base = torch.nn.Linear(7, 11)
    layer = LoRALinear(base, 3, 6, 0.1)
    assert tuple(layer.lora_A.shape) == (3, 7)
    assert tuple(layer.lora_B.shape) == (11, 3)
    assert tuple(layer(torch.zeros(2, 4, 7)).shape) == (2, 4, 11)

"""LoRA: freezing, zero-init output, gradients, adapter save/load, merging, and parity with PEFT."""
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


def test_matches_peft_forward_grad_update_merge():
    """Same weights in my LoRA and Hugging Face PEFT: outputs, loss, gradients, one SGD step and merged weights agree in float64."""
    pytest.importorskip('peft')
    from peft import LoraConfig
    from peft.tuners.lora import LoraModel
    torch.manual_seed(3)
    cfg = GPTConfig(31, 8, 1, 2, 8)
    base = GPT(cfg).double()
    custom = copy.deepcopy(base)
    reference = copy.deepcopy(base)
    targets = ('attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj')
    apply_lora(custom, rank=2, alpha=4, dropout=0, targets=targets)
    peft_model = LoraModel(reference, LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=list(targets), bias='none'), 'default')
    with torch.no_grad():
        for name, module in custom.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_B.normal_(std=0.03)
                p = peft_model.model.get_submodule(name)
                p.lora_A['default'].weight.copy_(module.lora_A)
                p.lora_B['default'].weight.copy_(module.lora_B)
    def trainable(model):
        return {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    assert sum(trainable(custom).values()) == sum(trainable(peft_model).values())
    assert len(trainable(custom)) == len(trainable(peft_model)) == 2 * len(targets) * cfg.n_layer
    idx = torch.randint(0, cfg.vocab_size, (2, 5))
    targets_y = torch.randint(0, cfg.vocab_size, (2, 5))
    out_custom, loss_custom = custom(idx, targets_y)
    out_peft, loss_peft = peft_model(idx, targets_y)
    torch.testing.assert_close(out_custom, out_peft, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(loss_custom, loss_peft, rtol=1e-11, atol=1e-11)
    loss_custom.backward(); loss_peft.backward()
    for name, module in custom.named_modules():
        if isinstance(module, LoRALinear):
            p = peft_model.model.get_submodule(name)
            torch.testing.assert_close(module.lora_A.grad, p.lora_A['default'].weight.grad, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(module.lora_B.grad, p.lora_B['default'].weight.grad, rtol=1e-10, atol=1e-10)
    opt1 = torch.optim.SGD((p for p in custom.parameters() if p.requires_grad), lr=0.01)
    opt2 = torch.optim.SGD((p for p in peft_model.parameters() if p.requires_grad), lr=0.01)
    opt1.step(); opt2.step()
    torch.testing.assert_close(custom(idx), peft_model(idx), rtol=1e-10, atol=1e-10)
    merge_lora(custom)
    peft_model.merge_adapter()
    torch.testing.assert_close(custom(idx), peft_model(idx), rtol=1e-10, atol=1e-10)

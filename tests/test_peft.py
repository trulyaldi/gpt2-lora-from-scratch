import copy
import pytest

torch = pytest.importorskip('torch')
peft = pytest.importorskip('peft')
from peft import LoraConfig
from peft.tuners.lora import LoraModel
from gpt_torch import GPT, GPTConfig
from lora import apply_lora, LoRALinear, merge_lora


def test_peft_forward_grad_update_merge():
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

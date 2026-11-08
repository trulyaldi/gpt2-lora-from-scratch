import pytest

torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
from gpt_numpy import GPT as NumpyGPT, CrossEntropyLoss
from gpt_torch import GPT, GPTConfig, copy_from_numpy


def test_numpy_port_forward_attention_loss_gradients():
    cfg = GPTConfig(19, 8, 2, 2, 8)
    np_model = NumpyGPT(**vars(cfg), seed=7)
    pt_model = copy_from_numpy(GPT(cfg).double(), np_model)
    idx = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    target = np.array([[2, 3, 4, 5], [3, 2, 1, 0]])
    x = torch.tensor(idx)
    logits, loss = pt_model(x, torch.tensor(target))
    np_logits = np_model.forward(idx)
    np.testing.assert_allclose(logits.detach().numpy(), np_logits, rtol=1e-11, atol=1e-11)
    np_loss = CrossEntropyLoss()
    expected_loss = np_loss.forward(np_logits, target)
    np.testing.assert_allclose(loss.item(), expected_loss, rtol=1e-11)
    captured = {}
    def capture(_, __, out):
        # A forward hook that returns a value replaces the module output, so return None.
        captured['attention'] = out.detach().numpy()
    handle = pt_model.h[0].attn.register_forward_hook(capture)
    pt_model(x)
    handle.remove()
    hidden = np_model.wte.params['W'][idx] + np_model.wpe.params['W'][np.arange(idx.shape[1])]
    expected_attn = np_model.h[0].attn.forward(np_model.h[0].ln_1.forward(hidden))
    np.testing.assert_allclose(captured['attention'], expected_attn, rtol=1e-11, atol=1e-11)
    loss.backward()
    np_model.zero_grad()
    np_model.backward(np_loss.backward())
    np_grads = {n: g for n, _, g in np_model.parameters()}
    checked = set()
    for name, module in pt_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            pairs = [(module.weight, '.W', True), (module.bias, '.b', False)]
        elif isinstance(module, torch.nn.Embedding):
            pairs = [(module.weight, '.W', False)]
        elif isinstance(module, torch.nn.LayerNorm):
            pairs = [(module.weight, '.g', False), (module.bias, '.b', False)]
        else:
            continue
        for param, suffix, transpose in pairs:
            expected = np_grads[name + suffix]
            np.testing.assert_allclose(param.grad.numpy(), expected.T if transpose else expected,
                                       rtol=1e-8, atol=1e-10, err_msg=name + suffix)
            checked.add(name + suffix)
    assert checked == set(np_grads), set(np_grads) ^ checked
    altered = x.clone(); altered[:, -1] = 8
    torch.testing.assert_close(pt_model(x)[:, :-1], pt_model(altered)[:, :-1])

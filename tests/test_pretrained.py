"""Pretrained GPT-2 small loaded into gpt_torch.GPT must match HuggingFace's GPT2LMHeadModel.

Runs only when the weights are already in the local HuggingFace cache (benchmark.py downloads them)."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
tiktoken = pytest.importorskip('tiktoken')
from huggingface_hub import hf_hub_download
from gpt_torch import load_pretrained_gpt2

REPO = 'openai-community/gpt2'


def test_pretrained_gpt2_matches_huggingface():
    try:
        path = hf_hub_download(REPO, 'model.safetensors', local_files_only=True)
        from transformers import GPT2LMHeadModel
        reference = GPT2LMHeadModel.from_pretrained(REPO, local_files_only=True, attn_implementation='eager')
    except Exception as exc:  # not cached locally
        pytest.skip(f'GPT-2 weights not in local cache: {exc}')
    ours = load_pretrained_gpt2(path).eval()
    reference.eval()
    ids = torch.tensor([tiktoken.get_encoding('gpt2').encode('First Citizen:\nBefore we proceed any further, hear me speak.')])
    with torch.no_grad():
        a, b = ours(ids), reference(ids).logits
    torch.testing.assert_close(a, b, rtol=1e-4, atol=5e-4)
    assert torch.equal(a.argmax(-1), b.argmax(-1))

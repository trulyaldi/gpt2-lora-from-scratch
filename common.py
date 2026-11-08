"""Shared helpers: the Tiny Shakespeare dataset and a description of the machine."""
import os
import platform
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'tinyshakespeare.txt'
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
HF_REPO = 'openai-community/gpt2'


def load_shakespeare(path=DATA_PATH):
    """Download Tiny Shakespeare if needed and return (train_tokens, val_tokens, n_characters).

    Text is tokenized with GPT-2's BPE (tiktoken). The split is contiguous: the last 10% of
    tokens is held out for validation."""
    import tiktoken
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    text = path.read_text(encoding='utf-8')
    tokens = np.array(tiktoken.get_encoding('gpt2').encode_ordinary(text), dtype=np.int64)
    cut = int(0.9 * len(tokens))
    return tokens[:cut], tokens[cut:], len(text)


def gpt2_weights_path():
    """Local path of the pretrained GPT-2 small safetensors file (downloaded once, ~550 MB)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, 'model.safetensors')


def system_info():
    """CPU, RAM, OS and Python version of this machine."""
    info = {'python': platform.python_version(), 'numpy': np.__version__, 'os': platform.platform(),
            'wsl2': 'microsoft' in platform.release().lower(), 'cpu_logical_cores': os.cpu_count()}
    try:
        with open('/proc/cpuinfo') as f:
            info['cpu'] = next(l.split(':', 1)[1].strip() for l in f if l.startswith('model name'))
        with open('/proc/meminfo') as f:
            info['ram_gib'] = round(next(int(l.split()[1]) for l in f if l.startswith('MemTotal')) / 2**20, 1)
    except (OSError, StopIteration):
        info['cpu'] = platform.processor()
    return info

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


SPLIT_FRACTIONS = {'train': 0.8, 'validation': 0.1, 'test': 0.1}


def read_shakespeare(path=DATA_PATH):
    """Tiny Shakespeare as a string, downloaded on first use."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding='utf-8')


def split_text(text, fractions=SPLIT_FRACTIONS):
    """Cut text into contiguous, non-overlapping sections in the given order and proportions.
    Each cut moves forward to the next blank line (speech boundary), so no speech straddles two
    splits; the sections concatenate back to the original text. Returns {name: (start, end)}."""
    if abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError('split fractions must sum to 1')
    bounds, start, acc = {}, 0, 0.0
    names = list(fractions)
    for name in names[:-1]:
        acc += fractions[name]
        cut = text.find('\n\n', int(acc * len(text)))
        cut = len(text) if cut < 0 else cut + 2
        bounds[name] = (start, cut)
        start = cut
    bounds[names[-1]] = (start, len(text))
    return bounds


def load_splits(names=('train', 'validation', 'test'), path=DATA_PATH):
    """GPT-2 BPE tokens of the requested splits plus metadata for all of them.

    The raw text is split into contiguous 80/10/10 sections first and each section is tokenized
    on its own, so no token and no training window can span two splits. Training code requests
    only ('train', 'validation'); the test split is loaded only by the final test evaluation."""
    import hashlib
    import tiktoken
    text = read_shakespeare(path)
    enc = tiktoken.get_encoding('gpt2')
    bounds = split_text(text)
    tokens, meta = {}, {'source': DATA_URL, 'characters': len(text),
                        'sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'tokenizer': 'GPT-2 BPE (tiktoken encode_ordinary), each split tokenized separately',
                        'splits': {}}
    for name, (start, end) in bounds.items():
        ids = np.array(enc.encode_ordinary(text[start:end]), dtype=np.int64)
        meta['splits'][name] = {'fraction': SPLIT_FRACTIONS[name], 'char_start': start, 'char_end': end,
                                'characters': end - start, 'tokens': len(ids)}
        if name in names:
            tokens[name] = ids
    return tokens, meta


def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device)."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

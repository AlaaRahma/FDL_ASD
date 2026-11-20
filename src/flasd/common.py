"""Repo-relative paths, constants, and small helpers."""
import json
import os
import random
import re
import unicodedata
from pathlib import Path

import numpy as np

# Package lives at <repo>/src/flasd/ → repo root is parents[2]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RUNS = ROOT / 'results'
CONFIGS = ROOT / 'configs'

# Local data layout (self-contained copies under data/)
AUTHENTIC_DIR = DATA / 'authentic'
AUTHENTIC = AUTHENTIC_DIR  # authentic-only train / val / test
SADA_CSV = DATA / 'sada' / 'sada.csv'  # preferred name (4,229)
SADA_TEACHER = DATA / 'sada' / 'sada_teacher_pens.csv'
SADA_TAGGED = DATA / 'sada' / 'sada_train_tagged.csv'
# Back-compat aliases used by older scripts / paper notes
SADA_FINAL = AUTHENTIC_DIR
SADA_FINAL_C = SADA_CSV
SADA_FINAL_C_TEACHER = SADA_TEACHER

# Optional legacy contaminated split (not shipped by default)
OLD_TRAIN = ROOT / 'data' / 'legacy' / 'sarcasm_train_80.csv'
OLD_TEST = ROOT / 'data' / 'legacy' / 'sarcasm_test_20.csv'

MODELS = {
    'MARBERTv2': 'UBC-NLP/MARBERTv2',
    'ARBERTv2': 'UBC-NLP/ARBERTv2',
    'AraBERTv02': 'aubmindlab/bert-base-arabertv02',
    'MARBERT': 'UBC-NLP/MARBERT',
}
SOURCES = ['Arabic_headlines', 'Arsarcasm_v2', 'Isarcasm']
SOURCE_TAGS = {
    'Arabic_headlines': 'خبر :',
    'Arsarcasm_v2': 'تغريدة أ :',
    'Isarcasm': 'تغريدة ب :',
}
SEEDS = (42, 43, 44)

# Default hyper-parameters (unified_v2 / thesis config)
HP = {
    'lr': {'full': 1e-5, 'lora': 2e-5, 'qlora': 2e-5},
    'batch_size': 64,
    'max_length': 64,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'warmup_ratio': 0.1,
    'label_smoothing': 0.1,
    'min_delta': 0.001,
    'epochs': 50,
    'patience': 10,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'lora_targets': ['query', 'key', 'value'],
}


def normalize_text(text):
    """NFKC; strip Arabic diacritics / tatweel / format characters; collapse whitespace."""
    text = unicodedata.normalize('NFKC', str(text))
    text = re.sub('[\u064b-\u065f\u0670\u0640]', '', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    return ' '.join(text.split())


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=_default)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)

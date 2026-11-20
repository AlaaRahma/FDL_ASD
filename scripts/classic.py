#!/usr/bin/env python3
"""TF-IDF / BiLSTM baselines. Thin wrapper around ``python -m flasd.classic``."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.argv[0] = 'flasd.classic'
runpy.run_module('flasd.classic', run_name='__main__')

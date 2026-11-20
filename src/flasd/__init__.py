"""FDL-ASD: federated LoRA Arabic sarcasm detection.

Modules
  data     — splits, tags, soft labels, client partitions
  models   — MARBERT / ARBERT / AraBERT full / LoRA / QLoRA
  train    — local optimisation loop (central + client)
  eval     — metrics and batched inference
  fl       — FedAvg / FedAvgM / FedDyn / …
  classic  — TF-IDF SVM / LR / CNB (+ optional BiLSTM)
  privacy  — membership-inference helpers
  stats    — paired bootstrap / Holm

CLIs live under ``scripts/`` (``run.py``, ``classic.py``).
"""

__version__ = '0.1.0'

"""Batched inference and evaluation metrics.

Validation/test selection rule used everywhere: pick the checkpoint by authentic
validation macro-F1 only; score the authentic test split once with that checkpoint.
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .common import HP


def amp_ctx(mode):
    if mode == 'qlora' or not torch.cuda.is_available():
        return torch.autocast('cpu', enabled=False)
    return torch.autocast('cuda', dtype=torch.float16)


@torch.no_grad()
def predict(model, data, mode, device, index=None, batch_size=256):
    model.eval()
    index = np.arange(len(data.y)) if index is None else np.asarray(index)
    order = index[np.argsort([len(data.ids[i]) for i in index], kind='stable')]
    probs = np.zeros((len(index), 2), dtype=np.float64)
    pos = {i: k for k, i in enumerate(index)}
    for chunk in [order[i:i + batch_size] for i in range(0, len(order), batch_size)]:
        ids, mask, _ = data.batch(chunk, device)
        with amp_ctx(mode):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        p = torch.softmax(logits, -1).cpu().numpy()
        for r, i in enumerate(chunk):
            probs[pos[i]] = p[r]
    return probs


def metrics(y, prob):
    y = np.asarray(y)
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    eps = HP['label_smoothing']
    q = np.eye(2)[y] * (1 - eps) + eps / 2
    out = {
        'macro_f1': float(f1_score(y, pred, average='macro')),
        'accuracy': float(accuracy_score(y, pred)),
        'precision_sarcastic': float(p[1]),
        'recall_sarcastic': float(r[1]),
        'f1_sarcastic': float(f[1]),
        'macro_precision': float(p.mean()),
        'macro_recall': float(r.mean()),
        'weighted_f1': float(f1_score(y, pred, average='weighted')),
        'loss_ce': float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean()),
        'loss_ls_ce': float(-(q * np.log(np.clip(prob, 1e-12, 1))).sum(1).mean()),
        'confusion': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if len(set(y)) == 2:
        scores = prob[:, 1]
        if np.isfinite(scores).all():
            out['auc_roc'] = float(roc_auc_score(y, scores))
            out['auc_pr'] = float(average_precision_score(y, scores))
        else:
            out['auc_roc'] = float('nan')
            out['auc_pr'] = float('nan')
    return out

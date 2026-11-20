"""Paired stratified bootstrap over test items (source x label strata), pooled over shared seeds, and Holm."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def load_preds(run_dir):
    return pd.read_csv(f'{run_dir}/test_predictions.csv')


def _f1(y, p):
    return f1_score(y, p, average='macro')


def paired_bootstrap(arm_preds, ref_preds, n_boot=5000, seed=0):
    """arm_preds / ref_preds: lists (same seed order) of DataFrames with id, source, label, prob_sarcastic.
    Statistic: mean over seeds of macro-F1(arm) - macro-F1(ref). Returns delta, CI low, CI high, two-sided p."""
    base = arm_preds[0][['id', 'source', 'label']].reset_index(drop=True)
    A = np.stack([(a.set_index('id').loc[base.id, 'prob_sarcastic'].values >= 0.5).astype(int) for a in arm_preds])
    B = np.stack([(b.set_index('id').loc[base.id, 'prob_sarcastic'].values >= 0.5).astype(int) for b in ref_preds])
    y = base.label.values
    delta = float(np.mean([_f1(y, a) - _f1(y, b) for a, b in zip(A, B)]))
    strata = [np.flatnonzero((base.source.values == s) & (y == c)) for s in base.source.unique() for c in (0, 1)]
    strata = [s for s in strata if len(s)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        yy = y[idx]
        boots[b] = np.mean([_f1(yy, a[idx]) - _f1(yy, r[idx]) for a, r in zip(A, B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return delta, float(lo), float(hi), float(max(p, 1.0 / n_boot))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj

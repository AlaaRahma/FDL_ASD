#!/usr/bin/env python3
"""Prior-work baselines re-run on the common split (authentic validation / test), CPU only.

  svm   LinearSVC on TF-IDF word 1-2 grams + char_wb 2-5 grams (Nayel et al. 2021 style)
  lr    logistic regression, same features
  cnb   complement naive Bayes, same features
  bilstm  2-layer BiLSTM over whitespace tokens with trainable embeddings (Abu Farha & Magdy 2020 style)
Hyper-parameters are chosen on validation macro-F1 only; the test split is scored once.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from . import data as D
from .common import RUNS, atomic_json, seed_everything
from .eval import metrics

# torch is only required for BiLSTM (and ``--help`` must work without it)
try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def prob_from_score(s):
    p = 1 / (1 + np.exp(-s))
    return np.c_[1 - p, p]


def tfidf_models(Xtr, ytr, Xva, yva, Xte, kind):
    grid = {'svm': [0.03, 0.1, 0.3, 1, 3], 'lr': [0.3, 1, 3, 10, 30], 'cnb': [0.03, 0.1, 0.3, 1]}[kind]
    best = (-1, None, None)
    for c in grid:
        if kind == 'svm':
            m = LinearSVC(C=c, class_weight='balanced', max_iter=20000).fit(Xtr, ytr)
            pv, pt = prob_from_score(m.decision_function(Xva)), prob_from_score(m.decision_function(Xte))
        elif kind == 'lr':
            m = LogisticRegression(C=c, class_weight='balanced', max_iter=5000).fit(Xtr, ytr)
            pv, pt = m.predict_proba(Xva), m.predict_proba(Xte)
        else:
            m = ComplementNB(alpha=c).fit(Xtr, ytr)
            pv, pt = m.predict_proba(Xva), m.predict_proba(Xte)
        f = metrics(yva, pv)['macro_f1']
        if f > best[0]:
            best = (f, c, (pv, pt))
    return best


def _bilstm_cls():
    if torch is None or nn is None:
        raise ImportError('torch is required for BiLSTM baselines; pip install -r requirements.txt')

    class BiLSTM(nn.Module):
        def __init__(self, vocab, emb=300, hid=128):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=0)
            self.lstm = nn.LSTM(emb, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.drop = nn.Dropout(0.5)
            self.out = nn.Linear(4 * hid, 2)

        def forward(self, x):
            h, _ = self.lstm(self.emb(x))
            m = (x > 0).unsqueeze(-1).float()
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            mx = (h - 1e4 * (1 - m)).max(1).values
            return self.out(self.drop(torch.cat([mean, mx], -1)))

    return BiLSTM


def bilstm(tr_t, ytr, va_t, yva, te_t, seed, epochs=20, patience=4):
    BiLSTM = _bilstm_cls()
    seed_everything(seed)
    from collections import Counter
    cnt = Counter(w for t in tr_t for w in t.split())
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= 2}
    enc = lambda ts: torch.tensor([[vocab.get(w, 1) for w in t.split()[:64]] + [0] * (64 - len(t.split()[:64]))
                                   for t in ts])
    Xtr, Xva, Xte = enc(tr_t), enc(va_t), enc(te_t)
    ytr_t = torch.tensor(ytr)
    model = BiLSTM(len(vocab) + 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    w = torch.tensor([1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=w)
    pred = lambda X: torch.softmax(torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)]), -1).numpy()
    best, stale, g = (-1, None, None, 0), 0, torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            lossf(model(Xtr[b]), ytr_t[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = pred(Xva)
        f = metrics(yva, pv)['macro_f1']
        if f > best[0] + 1e-3:
            with torch.no_grad():
                best, stale = (f, pv, pred(Xte), ep + 1), 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='auth')
    ap.add_argument('--methods', default='svm,lr,cnb,bilstm')
    ap.add_argument('--seeds', default='42,43,44')
    a = ap.parse_args()
    if torch is not None:
        torch.set_num_threads(8)
    elif 'bilstm' in a.methods.split(','):
        raise ImportError('torch is required for BiLSTM; omit it from --methods or pip install torch')
    train, val, test = D.load_splits(a.data)
    tr_t, va_t, te_t = D.model_text(train), D.model_text(val), D.model_text(test)
    ytr, yva, yte = train.sarcasm.values, val.sarcasm.values, test.sarcasm.values
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
    feats = None
    for method in a.methods.split(','):
        seeds = [int(s) for s in a.seeds.split(',')] if method == 'bilstm' else [42]
        for seed in seeds:
            out = RUNS / 'classic' / a.data / method / f'seed_{seed}'
            if (out / 'done.json').exists():
                continue
            t0 = time.time()
            if method == 'bilstm':
                f, pv, pt, ep = bilstm(tr_t, ytr, va_t, yva, te_t, seed)
                extra = {'selected': ep}
            else:
                if feats is None:
                    Xtr = hstack([vw.fit_transform(tr_t), vc.fit_transform(tr_t)]).tocsr()
                    feats = (Xtr, hstack([vw.transform(va_t), vc.transform(va_t)]).tocsr(),
                             hstack([vw.transform(te_t), vc.transform(te_t)]).tocsr())
                f, c, (pv, pt) = tfidf_models(feats[0], ytr, feats[1], yva, feats[2], method)
                extra = {'selected': c}
            res = {'status': 'done', 'method': method, 'data': a.data, 'seed': seed, **extra,
                   'best_val_macro_f1': f, 'val': metrics(yva, pv), 'test': metrics(yte, pt),
                   'test_by_source': {s: metrics(yte[test.source_file.values == s], pt[test.source_file.values == s])
                                      for s in sorted(set(test.source_file))},
                   'wall_seconds_total': time.time() - t0}
            if a.data == 'old':
                res['test_by_origin'] = {o: metrics(yte[test.origin.values == o], pt[test.origin.values == o])
                                         for o in ('real', 'synthetic')}
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'id': test.id.values, 'source': test.source_file.values, 'label': yte,
                          'prob_sarcastic': pt[:, 1]}).to_csv(out / 'test_predictions.csv', index=False)
            atomic_json(res, out / 'done.json')
            print(a.data, method, seed, 'val', round(f, 4), 'test', round(res['test']['macro_f1'], 4), flush=True)


if __name__ == '__main__':
    main()

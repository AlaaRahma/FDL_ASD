"""Membership-inference attacks (black-box, score-based) and a Renyi-DP accountant for client-level DP-FedAvg."""
import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _tpr_at(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def mia_scores(member_prob, member_y, nonmember_prob, nonmember_y):
    """Loss attack (Yeom et al. 2018), confidence attack and modified-entropy attack (Song & Mittal 2021).
    Members = training rows, non-members = test rows (never trained on). Balanced 1:1 when sizes match."""
    def per(prob, y):
        p_true = np.clip(prob[np.arange(len(y)), y], 1e-12, 1)
        p_other = np.clip(1 - p_true, 1e-12, 1)
        loss = -np.log(p_true)
        conf = prob.max(1)
        mentr = -((1 - p_true) * np.log(p_other) + p_other * np.log(p_true))
        return loss, conf, mentr
    lm, cm, em = per(member_prob, member_y)
    ln, cn, en = per(nonmember_prob, nonmember_y)
    y = np.r_[np.ones(len(lm)), np.zeros(len(ln))]
    out = {'n_members': int(len(lm)), 'n_nonmembers': int(len(ln)),
           'train_loss_mean': float(lm.mean()), 'test_loss_mean': float(ln.mean())}
    for name, s in (('loss', -np.r_[lm, ln]), ('confidence', np.r_[cm, cn]), ('mentropy', -np.r_[em, en])):
        out[f'{name}_auc'] = float(roc_auc_score(y, s))
        out[f'{name}_tpr_at_1fpr'] = _tpr_at(y, s, 0.01)
        out[f'{name}_tpr_at_01fpr'] = _tpr_at(y, s, 0.001)
    # loss-threshold attack accuracy with the threshold = mean training loss (Yeom et al.)
    thr = lm.mean()
    out['loss_threshold_advantage'] = float((lm <= thr).mean() - (ln <= thr).mean())
    return out


def rdp_gaussian_epsilon(noise_multiplier, rounds, delta, q=1.0):
    """(epsilon, delta) of `rounds` compositions of the (subsampled) Gaussian mechanism with sensitivity 1 at the
    client level. q = 1: exact RDP alpha / (2 sigma^2). q < 1: the small-q bound 2 q^2 alpha / sigma^2 (Mironov et
    al. 2019, valid for sigma >= 1 and moderate alpha); reported as an approximation."""
    best = math.inf
    for alpha in [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256)):
        rdp = alpha / (2 * noise_multiplier ** 2) if q >= 1 else 2 * q * q * alpha / noise_multiplier ** 2
        eps = rounds * rdp + math.log(1 / delta) / (alpha - 1)
        best = min(best, eps)
    return best

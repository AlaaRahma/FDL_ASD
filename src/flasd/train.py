"""Local training loop shared by centralised and federated runs."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import HP
from .eval import amp_ctx
from .models import get_state, set_state  # noqa: F401 — re-exported for callers


class Encoded:
    def __init__(self, tokenizer, texts, labels, max_length, soft=None, weight=None, origin=None):
        enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.ids = [np.asarray(x, dtype=np.int64) for x in enc['input_ids']]
        self.y = np.asarray(labels, dtype=np.int64)
        self.pad = tokenizer.pad_token_id
        self.soft = None if soft is None else np.asarray(soft, dtype=np.float64)
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float64)
        self.origin = None if origin is None else np.asarray(origin)

    def batch(self, idx, device):
        L = max(len(self.ids[i]) for i in idx)
        ids = np.full((len(idx), L), self.pad, dtype=np.int64)
        mask = np.zeros((len(idx), L), dtype=np.int64)
        for r, i in enumerate(idx):
            ids[r, :len(self.ids[i])] = self.ids[i]
            mask[r, :len(self.ids[i])] = 1
        return (
            torch.from_numpy(ids).to(device, non_blocking=True),
            torch.from_numpy(mask).to(device, non_blocking=True),
            torch.from_numpy(self.y[idx]).to(device, non_blocking=True),
        )


def batches(index, batch_size, rng, shuffle=True):
    index = np.asarray(index)
    if shuffle:
        index = index[rng.permutation(len(index))]
    return [index[i:i + batch_size] for i in range(0, len(index), batch_size)]


def lr_factor(progress):
    """Linear warm-up, then cosine decay to zero."""
    w = HP['warmup_ratio']
    if progress < w:
        return max(progress / w, 1e-3)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - w) / (1 - w))))


class Hooks:
    """Algorithm hooks inside a local training loop (no-ops = plain local SGD)."""

    def logits_adjust(self, logits, labels):
        return logits

    def extra_loss(self, model, ids, mask, rep):
        return None

    def on_grads(self, params):
        pass


def train_steps(model, params, data, index, mode, device, epochs, lr, lr_progress, rng, hooks=None, opt=None,
                scaler=None, capture=None, synchronize=True):
    """Run `epochs` passes over `index`. Returns (optimizer, scaler, stats)."""
    hooks = hooks or Hooks()
    model.train()
    if opt is None:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=HP['weight_decay'])
    use_scaler = mode != 'qlora' and torch.cuda.is_available()
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    plan = [b for _ in range(epochs) for b in batches(index, HP['batch_size'], rng)]
    n, losses, t0 = len(plan), [], time.time()
    for i, b in enumerate(plan):
        f = lr_factor(lr_progress(i, n))
        for g in opt.param_groups:
            g['lr'] = lr * f
        ids, mask, y = data.batch(b, device)
        with amp_ctx(mode):
            out = model(input_ids=ids, attention_mask=mask)
            logits = hooks.logits_adjust(out.logits.float(), y)
            if data.soft is not None or data.weight is not None:
                hard = F.cross_entropy(logits, y, reduction='none', label_smoothing=HP['label_smoothing'])
                if data.soft is not None:
                    soft = torch.as_tensor(data.soft[b], device=device, dtype=logits.dtype)
                    has = torch.isfinite(soft).all(dim=-1)
                    per = hard.clone()
                    if has.any():
                        soft_ce = -(soft[has].clamp_min(1e-12) * F.log_softmax(logits[has], -1)).sum(-1)
                        per = per.clone()
                        per[has] = soft_ce
                else:
                    per = hard
                if data.weight is not None:
                    w = torch.as_tensor(data.weight[b], device=device, dtype=per.dtype)
                    loss = (per * w).sum() / w.sum().clamp_min(1e-8)
                else:
                    loss = per.mean()
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=HP['label_smoothing'])
            extra = hooks.extra_loss(model, ids, mask, capture.get() if capture is not None else None)
            if extra is not None:
                loss = loss + extra
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        hooks.on_grads(params)
        torch.nn.utils.clip_grad_norm_(params, HP['grad_clip'])
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
    if synchronize and torch.cuda.is_available():
        torch.cuda.synchronize()
    return opt, scaler, {
        'steps': n,
        'train_loss': float(np.mean(losses)) if losses else float('nan'),
        'seconds': time.time() - t0,
        'examples': int(sum(len(b) for b in plan)),
    }


class RepCapture:
    """Captures pooled [CLS] representation entering the classification head (MOON)."""

    def __init__(self, model):
        self.value = None
        head = _classifier(model)
        head.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.value = inputs[0]

    def get(self):
        return self.value


def _classifier(model):
    for name, m in model.named_modules():
        if name.endswith('classifier') and not name.endswith('modules_to_save.default.classifier'):
            return m
    raise RuntimeError('no classifier module')

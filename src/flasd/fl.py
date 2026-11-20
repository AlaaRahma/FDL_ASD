"""Federated simulation (one process; clients can train in parallel on one GPU via model replicas).

Only the trainable parameters are exchanged (full mode: every parameter; LoRA / QLoRA: adapters + classification
head; the frozen backbone is shipped once and never changes, except under the 'merge' LoRA aggregation).

Algorithms (server x client rules), all with sample-size weights p_k = n_k / sum_{j in S_t} n_j:
  fedavg    McMahan et al. 2017
  fedprox   local gradient + mu (theta - theta_g)                                    Li et al. 2020
  scaffold  local gradient + c - c_k; c_k <- c_k - c - delta/(eta*tau) (Option II)   Karimireddy et al. 2020
  fedavgm   server momentum on the averaged update                                   Hsu et al. 2019
  fedadam / fedyogi / fedadagrad   adaptive server optimisers on the pseudo-gradient Reddi et al. 2021
  fednova   normalised averaging, tau_eff sum p_k Delta_k / tau_k                    Wang et al. 2020
  feddyn    dynamic regulariser (weighted form)                                      Acar et al. 2021
  moon      model-contrastive term on the pooled representation                      Li et al. 2021
  fedlc     label-distribution-calibrated logits z_c - tau n_{k,c}^(-1/4)            Zhang et al. 2022
LoRA aggregation (lora / qlora): ab (average A and B separately), ffa (A frozen, exact), svd (rank-r SVD of the
averaged product, FlexLoRA-style), merge (exact: merge sum p_k s B_k A_k into the backbone, re-initialise adapters;
equivalent to stacking, FLoRA-style; LoRA only).
DP: optional client-level DP-FedAvg (update clipping + Gaussian noise on the sum; McMahan et al. 2018).

Parallel clients (--client-workers W): keep W model replicas on the same GPU and train up to W selected clients
concurrently (ThreadPoolExecutor + per-replica CUDA streams). Default W=2 for LoRA on a 40GB A100 (~2.3 GiB peak
per replica). MOON stays sequential (shared contrastive models). Parallelism is wall-clock only; it does not change
the aggregation maths.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from .common import HP
from .train import Hooks, RepCapture, train_steps
from .eval import metrics, predict
from .models import get_state, lora_pairs, lora_scaling, set_state, trainable_names

DEFAULTS = {'mu': 0.01, 'server_lr_mult': 50.0, 'beta1': 0.9, 'beta2': 0.99, 'tau': 1e-3, 'momentum': 0.5,
            'feddyn_alpha': 0.01, 'moon_mu': 1.0, 'moon_temp': 0.5, 'fedlc_tau': 1.0}


class FLHooks(Hooks):
    def __init__(self):
        self.prox = None          # (mu, global_params)
        self.scaffold = None      # (c, c_k) lists aligned with params
        self.dyn = None           # (alpha, g_k, global_params)
        self.lc = None            # calibration vector (2,)
        self.moon = None          # (mu, temp, model_g, cap_g, model_p, cap_p)

    def logits_adjust(self, logits, labels):
        return logits if self.lc is None else logits - self.lc

    def extra_loss(self, model, ids, mask, rep):
        if self.moon is None:
            return None
        mu, temp, mg, cg, mp, cp = self.moon
        with torch.no_grad():
            mg(input_ids=ids, attention_mask=mask)
            mp(input_ids=ids, attention_mask=mask)
        z, zg, zp = rep.float(), cg.get().float(), cp.get().float()
        pos = F.cosine_similarity(z, zg, dim=-1) / temp
        neg = F.cosine_similarity(z, zp, dim=-1) / temp
        return mu * F.cross_entropy(torch.stack([pos, neg], 1), torch.zeros(len(z), dtype=torch.long, device=z.device))

    def on_grads(self, params):
        with torch.no_grad():
            if self.scaffold is not None:
                c, ck = self.scaffold
                for i, p in enumerate(params):
                    if p.grad is None:
                        continue
                    # Correction only; c_k is updated from (x-y)/(eta*tau) after the round (Option II).
                    # Do not accumulate p.grad into controls: AMP GradScaler skips inf steps but those
                    # grads would still poison a running mean and turn c into Inf by round 2.
                    corr = (c[i] - ck[i]).to(dtype=p.grad.dtype)
                    if torch.isfinite(corr).all():
                        p.grad.add_(corr)
            if self.prox is not None:
                mu, g = self.prox
                for i, p in enumerate(params):
                    if p.grad is not None:
                        p.grad.add_((mu * (p.float() - g[i])).to(p.grad.dtype))
            if self.dyn is not None:
                alpha, gk, g = self.dyn
                for i, p in enumerate(params):
                    if p.grad is not None:
                        p.grad.add_((alpha * (p.float() - g[i]) - gk[i]).to(p.grad.dtype))


def _vec_bytes(state):
    return int(sum(v.numel() for v in state.values()) * 4)


def _client_workers(args, mode, algo, device):
    """Default 2 for LoRA on CUDA; 1 for full/qlora/CPU/MOON. Explicit --client-workers overrides."""
    raw = getattr(args, 'client_workers', None)
    if raw is None:
        w = 2 if (mode == 'lora' and device.type == 'cuda') else 1
    else:
        w = max(1, int(raw))
    if algo == 'moon' or mode == 'qlora' or device.type != 'cuda':
        return 1
    return w


def _build_replicas(model, builder, n, names):
    """Reuse the primary model as replica 0; build n-1 extra copies sharing the same init seed path."""
    replicas = [model]
    if n <= 1:
        return replicas
    if builder is None:
        raise ValueError('client_workers>1 requires a model builder')
    for _ in range(n - 1):
        m = builder()
        # Ensure trainable name order matches (PEFT can rename modules).
        assert trainable_names(m) == names
        replicas.append(m)
    return replicas


def _local_update(model, names, glob, c_glob, data, local_idx, mode, device, args, algo, hp, r, R, k,
                  c_cli_k, dyn_g_k, moon_prev_k, mg, mp, cap_main, cap_g, cap_p, stream=None):
    """Run one client's local steps. Returns dict or None if empty client."""
    if len(local_idx) == 0:
        return None
    hooks = FLHooks()
    params_by_name = dict(model.named_parameters())
    plist = [params_by_name[n] for n in names]
    g_list = [glob[n] for n in names]
    ctx = torch.cuda.stream(stream) if stream is not None and device.type == 'cuda' else nullcontext()
    with ctx:
        set_state(model, glob)
        if algo == 'fedprox':
            hooks.prox = (hp['mu'], g_list)
        if algo == 'scaffold':
            if c_cli_k is None:
                c_cli_k = {n: torch.zeros_like(v, device='cpu') for n, v in glob.items()}
            ck = [c_cli_k[n].to(device) for n in names]
            hooks.scaffold = ([c_glob[n] for n in names], ck)
        if algo == 'feddyn':
            if dyn_g_k is None:
                dyn_g_k = {n: torch.zeros_like(v, device='cpu') for n, v in glob.items()}
            hooks.dyn = (hp['feddyn_alpha'], [dyn_g_k[n].to(device) for n in names], g_list)
        if algo == 'fedlc':
            cnt = np.bincount(data.y[local_idx], minlength=2).astype(float) + 1.0
            hooks.lc = torch.tensor(hp['fedlc_tau'] * cnt ** -0.25, device=device, dtype=torch.float32)
        if algo == 'moon':
            set_state(mg, glob)
            set_state(mp, moon_prev_k if moon_prev_k is not None else glob)
            hooks.moon = (hp['moon_mu'], hp['moon_temp'], mg, cap_g, mp, cap_p)
        crng = np.random.default_rng(args.seed * 100003 + r * 1009 + int(k))
        prog = lambda i, n, r=r: (r - 1 + (i + 1) / n) / R
        _, _, st = train_steps(model, plist, data, local_idx, mode, device, args.local_epochs, args.lr, prog, crng,
                               hooks=hooks, capture=cap_main, synchronize=False)
        local = get_state(model, names)
        if device.type == 'cuda' and stream is not None:
            stream.synchronize()
    delta = {n: local[n].clone().sub_(glob[n]) for n in names}
    out = {'k': int(k), 'steps': st['steps'], 'stats': st, 'delta': delta,
           'lora': {n: local[n].clone() for n in names if 'lora_' in n} if any('lora_' in n for n in names) else None}
    if algo == 'moon':
        out['moon_prev'] = {n: v.cpu() for n, v in local.items()}
    if algo == 'scaffold':
        # Option II: c_k^+ = c_k - c + (x - y)/(eta * tau) = c_k - c - delta/(eta * tau).
        # Uses weight deltas (finite even when AMP skips an overflow step) instead of mean grads.
        tau = max(1, int(st['steps']))
        inv = 1.0 / (float(args.lr) * tau)
        old_ck = {n: (c_cli_k[n].detach().cpu() if torch.is_tensor(c_cli_k[n]) else c_cli_k[n]) for n in names}
        new_ck = {n: (old_ck[n].to(device).float() - c_glob[n].float() - delta[n].float() * inv).cpu()
                  for n in names}
        out['new_ck'] = new_ck
        out['old_ck'] = old_ck
    if algo == 'feddyn':
        out['dyn_g'] = {n: (dyn_g_k[n].to(device) - hp['feddyn_alpha'] * delta[n]).cpu() for n in names}
    return out


def run_federated(model, mode, names, data, val, val_y, parts, args, device, log, builder=None):
    """Returns (best_state, best_extra, history, summary)."""
    algo, agg = args.algo, args.lora_agg
    hp = {**DEFAULTS, **{k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None}}
    K = len(parts)
    sizes = np.array([len(p) for p in parts], dtype=float)
    m = max(1, int(round(args.participation * K)))
    lr = args.lr
    rng_part = np.random.default_rng(args.seed + 101)
    workers = _client_workers(args, mode, algo, device)

    glob = get_state(model, names)                        # global trainable state (GPU, fp32)
    zeros = lambda: {n: torch.zeros_like(v) for n, v in glob.items()}
    server_m, server_v = zeros(), ({n: torch.full_like(v, hp['tau'] ** 2) for n, v in glob.items()})
    c_glob, c_cli = (zeros(), [None] * K) if algo == 'scaffold' else (None, None)
    dyn_h, dyn_g = (zeros(), [None] * K) if algo == 'feddyn' else (None, None)
    moon_prev = [None] * K
    if algo == 'moon':
        mg, mp = builder(), builder()
        for mm in (mg, mp):
            mm.eval()
        cap_main, cap_g, cap_p = RepCapture(model), RepCapture(mg), RepCapture(mp)
    else:
        mg = mp = cap_main = cap_g = cap_p = None
    pairs = lora_pairs(names) if mode in ('lora', 'qlora') else []
    if agg == 'ffa':
        pairs = []
    scale = lora_scaling(model) if mode in ('lora', 'qlora') else 1.0
    merged_base = {}
    pairs_all = base_mods = None
    if agg == 'merge':
        if mode != 'lora':
            raise ValueError('merge aggregation needs an unquantised backbone (LoRA mode)')
        pairs_all = lora_pairs(names)
        base_mods = {pre: model.get_submodule(pre) for pre, _, _ in pairs_all}

    replicas = _build_replicas(model, builder, workers, names)
    streams = ([torch.cuda.Stream(device=device) for _ in range(workers)]
               if device.type == 'cuda' and workers > 1 else [None] * workers)

    up_bytes = _vec_bytes(glob) * (2 if algo == 'scaffold' else 1)
    down_bytes = up_bytes
    history, best = [], {'score': -1, 'round': 0, 'state': None, 'extra': None}
    stale, t_start = 0, time.time()
    total_up = total_down = 0
    R = args.rounds
    auth_first = int(getattr(args, 'auth_first_rounds', 0) or 0)
    if auth_first and getattr(data, 'origin', None) is not None:
        auth_parts = [p[data.origin[p] == 'real'] for p in parts]
        auth_sizes = np.array([max(len(p), 1) for p in auth_parts], dtype=float)
    else:
        auth_parts, auth_sizes = parts, sizes
    if workers > 1:
        log(f'parallel clients: client_workers={workers} replicas={len(replicas)} '
            f'(~{2.3 * workers:.1f} GiB LoRA peak budget on shared GPU)')

    for r in range(1, R + 1):
        t_round = time.time()
        round_parts = auth_parts if (auth_first and r <= auth_first) else parts
        round_sizes = auth_sizes if (auth_first and r <= auth_first) else sizes
        sel = np.sort(rng_part.choice(K, size=m, replace=False)) if m < K else np.arange(K)
        w = round_sizes[sel] / round_sizes[sel].sum()
        acc = zeros()
        nova_acc = zeros() if algo == 'fednova' else None
        dc_acc = zeros() if algo == 'scaffold' else None
        lora_store = {} if agg in ('svd', 'merge') or (agg == 'ab' and pairs) else None
        taus, loc_stats, clip_norms = [], [], []

        def run_k(j_k, replica_i):
            j, k = j_k
            local_idx = round_parts[k]
            res = _local_update(
                model=replicas[replica_i], names=names, glob=glob, c_glob=c_glob, data=data,
                local_idx=local_idx, mode=mode, device=device, args=args, algo=algo, hp=hp, r=r, R=R, k=int(k),
                c_cli_k=c_cli[k] if c_cli is not None else None,
                dyn_g_k=dyn_g[k] if dyn_g is not None else None,
                moon_prev_k=moon_prev[k], mg=mg, mp=mp, cap_main=cap_main, cap_g=cap_g, cap_p=cap_p,
                stream=streams[replica_i])
            return j, res

        jobs = list(enumerate(sel))
        results = [None] * len(jobs)
        if workers == 1:
            for j, k in jobs:
                results[j] = run_k((j, k), 0)[1]
        else:
            # Waves of W: concurrent local training on separate replicas + CUDA streams (no global lock).
            for start in range(0, len(jobs), workers):
                wave = jobs[start:start + workers]
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    futs = [pool.submit(run_k, (j, k), i) for i, (j, k) in enumerate(wave)]
                    for fut in as_completed(futs):
                        j, res = fut.result()
                        results[j] = res
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)

        for j, res in enumerate(results):
            if res is None:
                continue
            k = res['k']
            st = res['stats']
            taus.append(st['steps'])
            loc_stats.append(st)
            if algo == 'moon':
                moon_prev[k] = res['moon_prev']
            if lora_store is not None:
                lora_store[int(k)] = (None, res['lora'])
            delta = res['delta']
            if args.dp_noise is not None:
                norm = torch.sqrt(sum((d.float() ** 2).sum() for d in delta.values()))
                clip_norms.append(float(norm))
                f = min(1.0, args.dp_clip / (float(norm) + 1e-12))
                for n in names:
                    delta[n].mul_(f)
            ww = float(w[j]) if args.dp_noise is None else 1.0 / m
            for n in names:
                acc[n].add_(delta[n], alpha=ww)
                if nova_acc is not None:
                    nova_acc[n].add_(delta[n], alpha=ww / st['steps'])
            if algo == 'scaffold':
                new_ck = res['new_ck']
                old_ck = res['old_ck']
                for n in names:
                    dc_acc[n].add_(new_ck[n].to(device) - old_ck[n].to(device), alpha=1.0 / K)
                c_cli[k] = new_ck
            if algo == 'feddyn':
                dyn_g[k] = res['dyn_g']
            if lora_store is not None:
                lora_store[int(k)] = (ww, lora_store[int(k)][1])
            total_up += up_bytes
            total_down += down_bytes
            del delta
        # ------------------------------------------------------------------ server update
        old_lora = ({n: glob[n].clone() for n in names if 'lora_' in n}
                    if (lora_store is not None and pairs and agg == 'ab') else None)
        if args.dp_noise is not None:
            sd = args.dp_noise * args.dp_clip / m
            gen = torch.Generator(device=device).manual_seed(args.seed * 7 + r)
            for n in names:
                acc[n].add_(torch.randn(acc[n].shape, generator=gen, device=device) * sd)
        if algo == 'fednova':
            tau_eff = float(np.dot(w, taus))
            upd = {n: nova_acc[n] * tau_eff for n in names}
        else:
            upd = acc
        agg_err = None
        if algo in ('fedavg', 'fedprox', 'scaffold', 'fednova', 'moon', 'fedlc'):
            for n in names:
                glob[n].add_(upd[n])
        elif algo == 'fedavgm':
            for n in names:
                server_m[n].mul_(hp['momentum']).add_(upd[n])
                glob[n].add_(server_m[n])
        elif algo in ('fedadam', 'fedyogi', 'fedadagrad'):
            slr = hp['server_lr_mult'] * lr
            for n in names:
                d = upd[n]
                server_m[n].mul_(hp['beta1']).add_(d, alpha=1 - hp['beta1'])
                d2 = d * d
                if algo == 'fedadam':
                    server_v[n].mul_(hp['beta2']).add_(d2, alpha=1 - hp['beta2'])
                elif algo == 'fedyogi':
                    server_v[n].sub_((1 - hp['beta2']) * d2 * torch.sign(server_v[n] - d2))
                else:
                    server_v[n].add_(d2)
                glob[n].add_(slr * server_m[n] / (server_v[n].sqrt() + hp['tau']))
        elif algo == 'feddyn':
            for n in names:
                dyn_h[n].sub_(upd[n], alpha=hp['feddyn_alpha'] * m / K)
                glob[n].add_(upd[n]).sub_(dyn_h[n] / hp['feddyn_alpha'])
        else:
            raise ValueError(algo)
        if algo == 'scaffold':
            for n in names:
                c_glob[n].add_(dc_acc[n])
        # ------------------------------------------------------------------ LoRA aggregation variants
        if lora_store is not None and pairs and agg in ('ab', 'svd', 'merge'):
            errs = []
            for pre, an, bn in (pairs if agg != 'merge' else pairs_all):
                prod = None
                for k, (ww, st) in lora_store.items():
                    pk = ww * (st[bn] @ st[an])
                    prod = pk if prod is None else prod + pk
                if agg == 'ab':
                    # share of the exact aggregated update sum_k p_k B_k A_k - B_g A_g lost by averaging A and B apart
                    naive = glob[bn] @ glob[an]
                    exact_upd = prod - old_lora[bn] @ old_lora[an]
                    errs.append(float(torch.linalg.norm(prod - naive) / (torch.linalg.norm(exact_upd) + 1e-12)))
                elif agg == 'svd':
                    rnk = glob[an].shape[0]
                    U, S, Vh = torch.linalg.svd(prod.float(), full_matrices=False)
                    sq = S[:rnk].sqrt()
                    glob[bn].copy_(U[:, :rnk] * sq)
                    glob[an].copy_(sq[:, None] * Vh[:rnk])
                elif agg == 'merge':
                    base = base_mods[pre].base_layer.weight
                    base.data.add_((scale * prod).to(base.dtype))
                    merged_base[pre] = base
                    torch.manual_seed(args.seed * 31 + r)
                    a = torch.empty_like(glob[an])
                    torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                    glob[an].copy_(a)
                    glob[bn].zero_()
            if errs:
                agg_err = float(np.mean(errs))
            if agg == 'merge':
                total_down += up_bytes * (m - 1) * m  # stacked adapters of all participants (FLoRA)
        # ------------------------------------------------------------------ evaluation (validation only)
        set_state(model, glob)
        vp = predict(model, val, mode, device)
        vm = metrics(val_y, vp)
        rec = {'round': r, 'val_macro_f1': vm['macro_f1'], 'val_accuracy': vm['accuracy'], 'val_loss': vm['loss_ce'],
               'val_auc': vm.get('auc_roc'), 'train_loss': float(np.average([s['train_loss'] for s in loc_stats],
                                                                            weights=[s['examples'] for s in loc_stats])),
               'clients': len(sel), 'local_steps': int(sum(taus)), 'seconds': time.time() - t_round,
               'comm_up_bytes_cum': total_up, 'comm_down_bytes_cum': total_down, 'agg_error': agg_err,
               'update_norm': float(torch.sqrt(sum((upd[n] ** 2).sum() for n in names))),
               'dp_clip_frac': float(np.mean([c > args.dp_clip for c in clip_norms])) if clip_norms else None,
               'client_workers': workers}
        history.append(rec)
        log(f"round {r}/{R} val_f1={vm['macro_f1']:.4f} loss={vm['loss_ce']:.4f} train={rec['train_loss']:.4f} "
            f"t={rec['seconds']:.0f}s w={workers}" + (f" aggerr={agg_err:.3f}" if agg_err is not None else ''))
        if vm['macro_f1'] > best['score'] + HP['min_delta']:
            best.update(score=vm['macro_f1'], round=r,
                        state={n: v.detach().cpu().clone() for n, v in glob.items()},
                        extra={pre: b.detach().cpu().clone() for pre, b in merged_base.items()})
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                log(f'early stop at round {r} (best round {best["round"]})')
                break
    summary = {'rounds_run': len(history), 'best_round': best['round'], 'best_val_macro_f1': best['score'],
               'upload_bytes_per_client_round': up_bytes, 'download_bytes_per_client_round': down_bytes,
               'comm_total_bytes': total_up + total_down,
               'comm_to_best_bytes': (history[best['round'] - 1]['comm_up_bytes_cum'] +
                                      history[best['round'] - 1]['comm_down_bytes_cum']) if best['round'] else None,
               'wall_seconds': time.time() - t_start, 'clients_per_round': m, 'hp': hp,
               'client_workers': workers}
    return best, history, summary


def restore_best(model, best):
    set_state(model, {n: v for n, v in best['state'].items()})
    for pre, w in (best.get('extra') or {}).items():
        model.get_submodule(pre).base_layer.weight.data.copy_(w.to(model.get_submodule(pre).base_layer.weight.device))

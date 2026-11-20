#!/usr/bin/env python3
"""One FDL-ASD run (centralised or federated). Writes <out>/done.json, history.csv, test_predictions.csv,
clients.csv, config.json, log.txt.

Examples
  run.py --setting central --model MARBERTv2 --mode lora --data auth --seed 42 --out runs/x
  run.py --setting fl --algo fedprox --partition dir0.1 --clients 10 --model MARBERTv2 --mode lora --data sada ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from flasd.common import HP, SADA_TEACHER, SEEDS, atomic_json, seed_everything  # noqa: E402
from flasd import data as D  # noqa: E402
from flasd.train import Encoded, lr_factor, train_steps  # noqa: E402
from flasd.eval import metrics, predict  # noqa: E402
from flasd.models import build, count_params, get_state, set_state, tokenizer_for, trainable_names  # noqa: E402
from flasd.privacy import mia_scores  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--setting', choices=['central', 'fl'], required=True)
    p.add_argument('--model', default='MARBERTv2')
    p.add_argument('--mode', choices=['full', 'lora', 'qlora'], default='lora')
    p.add_argument('--data', choices=['auth', 'sada', 'sadac', 'ros', 'old'], default='auth',
                   help='auth=18,548 authentic; sada/sadac=auth+SADA(4,229); ros=oversample matched to SADA')
    p.add_argument('--tag', action='store_true', help='source-tag prefix (domain prompt)')
    p.add_argument('--syn-marker', action='store_true', help='prepend [SYN] to synthetic train rows only')
    p.add_argument('--p2', action='store_true',
                   help='P2-style: --tag + --syn-marker + soft labels (p_ens) + synthetic-scale 0.3; '
                        'FL also uses --auth-first-rounds (default 10) as a curriculum approximation')
    p.add_argument('--synthetic-scale', type=float, default=None,
                   help='per-example weight for synthetic rows (P2 default 0.3)')
    p.add_argument('--teacher-logits', default=None,
                   help='CSV with id,logit_0,logit_1 soft labels for synthetic rows')
    p.add_argument('--auth-first-rounds', type=int, default=None,
                   help='FL curriculum approx: first N rounds train on authentic rows only')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', '--output', dest='out', required=True)
    p.add_argument('--epochs', type=int, default=HP['epochs'])
    p.add_argument('--patience', type=int, default=HP['patience'])
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--lora-r', type=int, default=None)
    # federated
    p.add_argument('--algo', default='fedavg',
                   choices=['fedavg', 'fedprox', 'scaffold', 'fedavgm', 'fedadam', 'fedyogi', 'fedadagrad', 'fednova',
                            'feddyn', 'moon', 'fedlc'])
    p.add_argument('--partition', default='natural')
    p.add_argument('--clients', type=int, default=10)
    p.add_argument('--participation', type=float, default=1.0)
    p.add_argument('--rounds', type=int, default=None)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--lora-agg', choices=['ab', 'ffa', 'svd', 'merge'], default='ab')
    p.add_argument('--client-workers', type=int, default=2,
                   help='within-round parallel local trainers (separate model replicas + CUDA streams). '
                        'Default 2 (LoRA-safe on A100-40GB, ~2.3 GiB/replica extra). Use 1 for sequential. '
                        'Forced to 1 for QLoRA / MOON / CPU. Full FT: prefer 1.')
    for k in ('mu', 'server_lr_mult', 'momentum', 'feddyn_alpha', 'moon_mu', 'moon_temp', 'fedlc_tau', 'beta1', 'beta2',
              'tau'):
        p.add_argument('--' + k.replace('_', '-'), dest=k, type=float, default=None)
    p.add_argument('--dp-noise', type=float, default=None, help='client-level DP-FedAvg noise multiplier')
    p.add_argument('--dp-clip', type=float, default=0.05)
    p.add_argument('--no-mia', action='store_true')
    p.add_argument('--save-model', action='store_true')
    p.add_argument('--max-train', type=int, default=None, help='smoke tests only: subsample training rows')
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists():
        print('already done', out)
        return
    logf = open(out / 'log.txt', 'a')

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    if args.p2:
        args.tag = True
        args.syn_marker = True
        if args.synthetic_scale is None:
            args.synthetic_scale = 0.3
        if args.teacher_logits is None:
            args.teacher_logits = str(SADA_TEACHER)
        if args.setting == 'fl' and args.auth_first_rounds is None:
            args.auth_first_rounds = 10
    args.lr = args.lr or HP['lr'][args.mode]
    if args.rounds is None:
        args.rounds = 30 if args.mode == 'full' else 50
    t0 = time.time()
    train, val, test = D.load_splits(args.data)
    if args.max_train:
        train = train.sample(n=min(args.max_train, len(train)), random_state=0).reset_index(drop=True)
        val, test = val.iloc[:500].reset_index(drop=True), test.iloc[:500].reset_index(drop=True)
    tok = tokenizer_for(args.model)
    soft = D.soft_targets_from_teacher(train, args.teacher_logits) if args.teacher_logits else None
    weight = D.sample_weights(train, args.synthetic_scale) if args.synthetic_scale is not None else None

    def enc(frame, with_soft=False):
        texts = D.model_text(frame, tag=args.tag, syn_marker=args.syn_marker and 'origin' in frame.columns)
        return Encoded(tok, texts, frame.sarcasm.values, HP['max_length'],
                       soft=soft if with_soft else None,
                       weight=weight if with_soft else None,
                       origin=frame.origin.values if 'origin' in frame.columns else None)

    tr_e, va_e, te_e = enc(train, with_soft=True), enc(val), enc(test)
    if args.setting == 'central':
        parts = [np.arange(len(train))]
        names_c = ['central']
    else:
        parts = D.partition(train, args.partition, args.clients, args.seed)
        names_c = D.client_names(train, args.partition, parts)
        parts = D.allocate_synthetic(train, parts, args.seed)
    clients = D.describe_clients(train, parts, names_c)
    pd.DataFrame(clients).to_csv(out / 'clients.csv', index=False)
    builder = lambda: build(args.model, args.mode, device, args.lora_r, freeze_a=args.lora_agg == 'ffa',
                            lora_seed=args.seed)
    model = builder()
    names = trainable_names(model)
    n_total, n_train = count_params(model)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    cfg = {**vars(args), 'n_train_rows': len(train), 'n_synthetic_rows': int((train.origin == 'synthetic').sum()),
           'n_val': len(val), 'n_test': len(test), 'params_total': n_total, 'params_trainable': n_train,
           'hp': HP, 'n_clients_effective': len(parts), 'torch': torch.__version__}
    if args.p2:
        cfg['p2_note'] = (
            'FL/central P2 approximation in fl-asd-paper: source tag on all texts; [SYN] on synthetic train only; '
            'soft labels from data/sada/sada_teacher_pens.csv; synthetic_scale=0.3; '
            'FL curriculum approx = auth-first rounds then full mix (not the centralised two-stage utag→synben curriculum). '
            'Centralised full-curriculum P2 results are reused from dataset_benchmark/sadac (and arsfix D1.5-w0.3).'
        )
    atomic_json(cfg, out / 'config.json')
    log(f'{args.setting} {args.model} {args.mode} data={args.data} tag={args.tag} syn={args.syn_marker} '
        f'p2={args.p2} scale={args.synthetic_scale} auth_first={args.auth_first_rounds} rows={len(train)} '
        f'trainable={n_train:,}/{n_total:,} clients={len(parts)}')

    if args.setting == 'central':
        best, history, summary = central(model, names, tr_e, va_e, val.sarcasm.values, args, device, log)
        set_state(model, best['state'])
    else:
        from flasd.fl import restore_best, run_federated
        best, history, summary = run_federated(model, args.mode, names, tr_e, va_e, val.sarcasm.values, parts, args,
                                               device, log, builder=builder)
        restore_best(model, best)
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == 'cuda' else 0.0

    # ------------------------------------------------------------------ single test evaluation
    vp = predict(model, va_e, args.mode, device)
    tp = predict(model, te_e, args.mode, device)
    res = {'status': 'done', 'val': metrics(val.sarcasm.values, vp), 'test': metrics(test.sarcasm.values, tp)}
    if abs(res['val']['macro_f1'] - best['score']) > 2e-3:
        log(f'WARNING reload check: val {res["val"]["macro_f1"]:.4f} vs selected {best["score"]:.4f}')
    res['reload_check_abs_diff'] = abs(res['val']['macro_f1'] - best['score'])
    res['test_by_source'] = {s: metrics(test.sarcasm.values[test.source_file.values == s],
                                        tp[test.source_file.values == s])
                             for s in sorted(set(test.source_file))}
    if args.data == 'old':
        res['test_by_origin'] = {o: metrics(test.sarcasm.values[test.origin.values == o], tp[test.origin.values == o])
                                 for o in ('real', 'synthetic')}
    pd.DataFrame({'id': test.id.values, 'source': test.source_file.values, 'label': test.sarcasm.values,
                  'prob_sarcastic': tp[:, 1]}).to_csv(out / 'test_predictions.csv', index=False)
    pd.DataFrame({'id': val.id.values, 'source': val.source_file.values, 'label': val.sarcasm.values,
                  'prob_sarcastic': vp[:, 1]}).to_csv(out / 'val_predictions.csv', index=False)

    # ------------------------------------------------------------------ privacy: membership inference
    if not args.no_mia and args.data != 'old':
        rng = np.random.default_rng(args.seed + 5)
        real = np.flatnonzero(train.origin.values == 'real')
        mem = rng.choice(real, size=min(len(test), len(real)), replace=False)
        mp = predict(model, tr_e, args.mode, device, index=mem)
        res['mia_global'] = mia_scores(mp, tr_e.y[mem], tp, te_e.y)
        if args.setting == 'fl':
            res['mia_server_view'] = server_view_mia(model, names, tr_e, te_e, tp, train, parts, args, device, best)
    res.update(summary)
    res.update({'params_total': n_total, 'params_trainable': n_train,
                'trainable_share': n_train / n_total, 'peak_gpu_gib': peak, 'wall_seconds_total': time.time() - t0,
                'selected': best.get('round') or best.get('epoch'), 'best_val_macro_f1': best['score']})
    if args.save_model:
        torch.save({n: v for n, v in best['state'].items()}, out / 'trainable_state.pt')
    atomic_json(res, out / 'done.json')
    log(f'DONE test macro-F1 {res["test"]["macro_f1"]:.4f} (val {best["score"]:.4f}, selected '
        f'{res["selected"]}) peak {peak:.1f} GiB')


def central(model, names, data, val, val_y, args, device, log):
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    idx = np.arange(len(data.y))
    steps_per_epoch = int(np.ceil(len(idx) / HP['batch_size']))
    total = steps_per_epoch * args.epochs
    opt = scaler = None
    best, history, stale, t0 = {'score': -1, 'epoch': 0, 'state': None}, [], 0, time.time()
    for ep in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed * 1000 + ep)
        prog = lambda i, n, ep=ep: ((ep - 1) * steps_per_epoch + i + 1) / total
        opt, scaler, st = train_steps(model, params, data, idx, args.mode, device, 1, args.lr, prog, rng, opt=opt,
                                      scaler=scaler)
        vm = metrics(val_y, predict(model, val, args.mode, device))
        history.append({'epoch': ep, 'val_macro_f1': vm['macro_f1'], 'val_accuracy': vm['accuracy'],
                        'val_loss': vm['loss_ce'], 'val_auc': vm.get('auc_roc'), 'train_loss': st['train_loss'],
                        'seconds': st['seconds'], 'lr_end': opt.param_groups[0]['lr']})
        log(f'epoch {ep} val_f1={vm["macro_f1"]:.4f} loss={vm["loss_ce"]:.4f} train={st["train_loss"]:.4f} '
            f't={st["seconds"]:.0f}s')
        if vm['macro_f1'] > best['score'] + HP['min_delta']:
            best = {'score': vm['macro_f1'], 'epoch': ep, 'state': get_state(model, names, device='cpu')}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                log(f'early stop at epoch {ep} (best {best["epoch"]})')
                break
    summary = {'epochs_run': len(history), 'best_epoch': best['epoch'], 'best_val_macro_f1': best['score'],
               'wall_seconds': time.time() - t0, 'seconds_per_epoch': float(np.mean([h['seconds'] for h in history]))}
    return best, history, summary


def server_view_mia(model, names, tr_e, te_e, tp, train, parts, args, device, best):
    """Honest-but-curious server: it receives the locally trained model of a client. Train one more local round of
    the largest client from the selected global model and attack that client's own authentic rows."""
    k = int(np.argmax([len(p) for p in parts]))
    own = parts[k][train.origin.values[parts[k]] == 'real']
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    rng = np.random.default_rng(args.seed + 77)
    sel = best.get('round') or 1
    prog = lambda i, n: min(1.0, (sel - 1 + (i + 1) / n) / args.rounds)
    train_steps(model, params, tr_e, parts[k], args.mode, device, args.local_epochs, args.lr, prog, rng)
    mem = rng.choice(own, size=min(len(own), len(te_e.y)), replace=False)
    mp = predict(model, tr_e, args.mode, device, index=mem)
    tp_local = predict(model, te_e, args.mode, device)
    out = mia_scores(mp, tr_e.y[mem], tp_local, te_e.y)
    out['client_rows'] = int(len(parts[k]))
    return out


if __name__ == '__main__':
    main()

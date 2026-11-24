"""Data variants, client partitions and allocation of synthetic rows to clients.

Validation and test are always authentic. Synthetic rows enter training only.

Variants:
  auth   authentic training split (18,548 rows)
  sada   auth + SADA (4,229) — paper protocol
  sadac  alias for ``sada`` (legacy CLI name)
  ros    auth + random oversampling of authentic sarcastic rows, matched to SADA strata
  old    original FDL-ASD split (legacy audit only; not shipped by default)
"""
import numpy as np
import pandas as pd

from .common import (AUTHENTIC, OLD_TEST, OLD_TRAIN, SADA_CSV, SOURCE_TAGS, normalize_text)

# dialect groups used by the natural partition (the synthetic ArSarcasm-v2 rows carry the merged label
# 'Maghreb+Nile Basin', so Maghreb and Nile Basin are one group for that source)
_GROUP = {
    ('Arsarcasm_v2', 'MSA'): 'ArS-MSA', ('Arsarcasm_v2', 'Nile Basin'): 'ArS-Nile+Maghreb',
    ('Arsarcasm_v2', 'Maghreb'): 'ArS-Nile+Maghreb', ('Arsarcasm_v2', 'Maghreb+Nile Basin'): 'ArS-Nile+Maghreb',
    ('Arsarcasm_v2', 'Gulf'): 'ArS-Gulf', ('Arsarcasm_v2', 'Levant'): 'ArS-Levant',
    ('Isarcasm', 'Nile Basin'): 'iSar-Nile', ('Isarcasm', 'MSA'): 'iSar-MSA', ('Isarcasm', 'Maghreb'): 'iSar-Maghreb',
    ('Isarcasm', 'Gulf'): 'iSar-Gulf+Levant', ('Isarcasm', 'Levant'): 'iSar-Gulf+Levant',
}


def stratum(source, dialect):
    if source == 'Arabic_headlines':
        return 'Headlines'
    return _GROUP.get((source, dialect), 'ArS-MSA' if source == 'Arsarcasm_v2' else 'iSar-MSA')


def _prep(frame, origin):
    frame = frame.copy()
    frame['id'] = frame['id'].astype(str)
    frame['sarcasm'] = frame['sarcasm'].astype(int)
    frame['origin'] = origin
    if 'dialect_region' not in frame:
        frame['dialect_region'] = 'NA'
    frame['dialect_region'] = frame['dialect_region'].fillna('NA')
    frame['group'] = [stratum(s, d) if s in SOURCE_TAGS else s for s, d in zip(frame.source_file, frame.dialect_region)]
    return frame[['id', 'text', 'sarcasm', 'source_file', 'dialect_region', 'origin', 'group']]


def _load_authentic():
    """Authentic-only splits (train must not contain the old 7,390 synthetic pool)."""
    train = pd.read_csv(AUTHENTIC / 'train.csv')
    if 'origin' in train.columns:
        train = train[train.origin.astype(str).str.lower().isin(['real', 'authentic', 'human'])]
    auth = _prep(train, 'real')
    val = _prep(pd.read_csv(AUTHENTIC / 'validation.csv'), 'real')
    test = _prep(pd.read_csv(AUTHENTIC / 'test.csv'), 'real')
    return auth, val, test


def _load_sada():
    path = SADA_CSV
    if not path.exists():
        # legacy filename fallback
        alt = path.with_name('sada_final_c.csv')
        path = alt if alt.exists() else path
    return pd.read_csv(path)


def load_splits(variant):
    """Return train (with column origin real/synthetic), validation, test frames (raw text)."""
    if variant == 'old':
        tr, te = pd.read_csv(OLD_TRAIN), pd.read_csv(OLD_TEST)
        for f in (tr, te):
            f['id'] = [f'old{i}' for i in range(len(f))]
            f['dialect_region'] = 'NA'
        org = lambda f: np.where(f.source_file.str.startswith('synthetic'), 'synthetic', 'real')
        tr, te = _old(tr), _old(te)
        tr['is_gpt4'], te['origin'] = org(tr) == 'synthetic', org(te)
        tr['origin'] = 'real'  # old protocol: GPT-4 rows are ordinary training rows (partitioned like the rest)
        val = pd.read_csv(AUTHENTIC / 'validation.csv')
        return tr, _prep(val, 'real'), te
    auth, val, test = _load_authentic()
    if variant == 'auth':
        train = auth
    elif variant in ('sada', 'sadac'):
        syn = _load_sada()
        train = pd.concat([auth, _prep(syn, 'synthetic')], ignore_index=True)
    elif variant == 'ros':
        syn = _load_sada()
        rng = np.random.default_rng(20261108)
        rows = []
        for (src, dia), cnt in syn.groupby(['source_file', 'dialect_region']).size().items():
            g = stratum(src, dia)
            pool = auth[(auth.group == g) & (auth.sarcasm == 1)]
            if len(pool) == 0:
                pool = auth[auth.sarcasm == 1]
            take = pool.iloc[rng.integers(0, len(pool), cnt)].copy()
            take['id'] = [f'ros_{i}_{x}' for i, x in enumerate(take.id)]
            rows.append(take)
        dup = pd.concat(rows, ignore_index=True)
        dup['origin'] = 'synthetic'  # duplicated rows are allocated to clients exactly like synthetic rows
        train = pd.concat([auth, dup], ignore_index=True)
    else:
        raise ValueError(variant)
    return train.reset_index(drop=True), val, test


def _old(f):
    src = {'cleaned_Arsarcasm1_modified.csv': 'Arsarcasm_v2', 'cleaned_Arsarcasm2_modified.csv': 'Arsarcasm_v2',
           'cleaned_Isarcasm_modified.csv': 'Isarcasm', 'cleaned_headlines_modified.csv': 'Arabic_headlines'}
    f = f.copy()
    f['orig_source'] = f.source_file
    f['source_file'] = [src.get(s, s) for s in f.source_file]
    f['group'] = f['source_file']
    f['sarcasm'] = f.sarcasm.astype(int)
    return f


def model_text(frame, tag=False, syn_marker=False):
    """Build model inputs. Source tag applies to every row (train/val/test). [SYN] applies only to synthetic train."""
    texts = [normalize_text(t) for t in frame.text]
    if tag:
        texts = [f'{SOURCE_TAGS.get(s, "")} {t}'.strip() for s, t in zip(frame.source_file, texts)]
    if syn_marker and 'origin' in frame.columns:
        texts = [f'[SYN] {t}' if o == 'synthetic' else t for t, o in zip(texts, frame.origin)]
    return texts


def soft_targets_from_teacher(train, teacher_csv):
    """Return (N, 2) soft targets for synthetic rows from teacher logits [log(1-p), log p]; NaN on authentic."""
    import pandas as pd
    soft = np.full((len(train), 2), np.nan, dtype=np.float64)
    if teacher_csv is None:
        return soft
    teach = pd.read_csv(teacher_csv)
    logits = teach.set_index(teach['id'].astype(str))[['logit_0', 'logit_1']]
    syn = train.origin.values == 'synthetic'
    missing = 0
    for i in np.flatnonzero(syn):
        key = str(train.id.values[i])
        if key not in logits.index:
            missing += 1
            continue
        row = logits.loc[key]
        # teacher columns are log(1-p_ens), log(p_ens); renormalise for safety
        p1 = float(np.exp(float(row['logit_1'])))
        p0 = float(np.exp(float(row['logit_0'])))
        z = p0 + p1
        soft[i] = (p0 / z, p1 / z)
    if missing:
        raise KeyError(f'{missing} synthetic ids missing from teacher logits {teacher_csv}')
    return soft


def sample_weights(train, synthetic_scale=1.0):
    w = np.ones(len(train), dtype=np.float64)
    w[train.origin.values == 'synthetic'] = float(synthetic_scale)
    return w


# ----------------------------------------------------------------------------------------------- partitions
def partition(train, scheme, n_clients, seed, min_size=40):
    """Client index arrays over the AUTHENTIC rows of train; synthetic rows are added by allocate_synthetic."""
    real = np.flatnonzero(train.origin.values == 'real')
    y = train.sarcasm.values
    rng = np.random.default_rng(seed)
    if scheme == 'central':
        parts = [real]
    elif scheme == 'iid':
        parts = [[] for _ in range(n_clients)]
        for key, idx in train.iloc[real].groupby(['source_file', 'sarcasm']).indices.items():
            idx = real[idx]
            rng.shuffle(idx)
            off = rng.integers(n_clients)
            for j, i in enumerate(idx):
                parts[(j + off) % n_clients].append(i)
        parts = [np.array(p) for p in parts]
    elif scheme.startswith('dir'):
        alpha = float(scheme[3:])
        for _ in range(1000):
            parts = [[] for _ in range(n_clients)]
            for c in (0, 1):
                idx = real[y[real] == c].copy()
                rng.shuffle(idx)
                p = rng.dirichlet(alpha * np.ones(n_clients))
                cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(idx, cuts)):
                    parts[k].extend(chunk.tolist())
            if min(len(p) for p in parts) >= min_size:
                break
        parts = [np.array(sorted(p)) for p in parts]
    elif scheme == 'source':
        parts = [real[train.source_file.values[real] == s] for s in sorted(set(train.source_file.values[real]))]
    elif scheme == 'natural':
        groups = train.group.values
        names = sorted(set(groups[real]))
        parts = []
        for g in names:
            idx = real[groups[real] == g]
            if g == 'Headlines':  # two news outlets x label structure: split into two stratified halves
                idx = idx.copy()
                rng.shuffle(idx)
                yy = y[idx]
                h0 = np.concatenate([idx[yy == 0][::2], idx[yy == 1][::2]])
                h1 = np.setdiff1d(idx, h0)
                parts += [np.sort(h0), np.sort(h1)]
            else:
                parts.append(idx)
    elif scheme == 'natural_balanced':
        # Same source×dialect strata as `natural`, but merge tiny clients and split huge ones so authentic
        # sizes stay within ~2× (K≈6). Story: Headlines (2), ArSarcasm-MSA (2), ArSarcasm-regional (1),
        # iSarcasm (1). n_clients is ignored (structure-determined), like natural.
        parts, _ = _natural_balanced_parts(train, real, y, rng)
    else:
        raise ValueError(scheme)
    return [np.asarray(p, dtype=int) for p in parts]


def _stratified_halves(idx, y, rng):
    idx = np.asarray(idx).copy()
    rng.shuffle(idx)
    yy = y[idx]
    h0 = np.concatenate([idx[yy == 0][::2], idx[yy == 1][::2]])
    h1 = np.setdiff1d(idx, h0)
    return np.sort(h0), np.sort(h1)


def _natural_balanced_parts(train, real, y, rng):
    """Return (parts, names) for the balanced natural partition over authentic rows."""
    groups = train.group.values
    by_g = {g: real[groups[real] == g] for g in sorted(set(groups[real]))}
    # Merge map: tiny / same-source siblings → one federated client; large kept for splitting.
    merge = {
        'Headlines': 'Headlines',
        'ArS-MSA': 'ArS-MSA',
        'ArS-Gulf': 'ArS-regional', 'ArS-Levant': 'ArS-regional', 'ArS-Nile+Maghreb': 'ArS-regional',
        'iSar-Maghreb': 'iSarcasm', 'iSar-Gulf+Levant': 'iSarcasm', 'iSar-MSA': 'iSarcasm', 'iSar-Nile': 'iSarcasm',
    }
    buckets = {}
    for g, idx in by_g.items():
        key = merge.get(g, g)
        buckets.setdefault(key, []).append(idx)
    buckets = {k: np.sort(np.concatenate(v)) for k, v in buckets.items()}

    parts, names = [], []
    # Fixed order for reproducibility / paper tables.
    order = ['Headlines', 'ArS-MSA', 'ArS-regional', 'iSarcasm']
    for key in order:
        if key not in buckets:
            continue
        idx = buckets[key]
        if key in ('Headlines', 'ArS-MSA'):
            a, b = _stratified_halves(idx, y, rng)
            parts += [a, b]
            names += [f'{key}-1', f'{key}-2']
        else:
            parts.append(idx)
            names.append(key)
    # Any unexpected residual group (should not happen): append as its own client.
    for key, idx in buckets.items():
        if key not in order:
            parts.append(idx)
            names.append(key)
    return parts, names


def client_names(train, scheme, parts):
    if scheme == 'natural':
        names = []
        for p in parts:
            g = train.group.values[p[0]]
            names.append(g if g != 'Headlines' else f'Headlines-{sum(n.startswith("Headlines") for n in names) + 1}')
        return names
    if scheme == 'natural_balanced':
        merge = {
            'Headlines': 'Headlines', 'ArS-MSA': 'ArS-MSA',
            'ArS-Gulf': 'ArS-regional', 'ArS-Levant': 'ArS-regional', 'ArS-Nile+Maghreb': 'ArS-regional',
            'iSar-Maghreb': 'iSarcasm', 'iSar-Gulf+Levant': 'iSarcasm', 'iSar-MSA': 'iSarcasm', 'iSar-Nile': 'iSarcasm',
        }
        names, seen = [], {}
        groups, origin = train.group.values, train.origin.values
        for p in parts:
            auth = p[origin[p] == 'real'] if origin is not None else p
            g = groups[auth[0] if len(auth) else p[0]]
            key = merge.get(g, g)
            if key in ('Headlines', 'ArS-MSA'):
                seen[key] = seen.get(key, 0) + 1
                names.append(f'{key}-{seen[key]}')
            else:
                names.append(key)
        return names
    if scheme == 'source':
        return [train.source_file.values[p[0]] for p in parts]
    return [f'client{k + 1}' for k in range(len(parts))]


def allocate_synthetic(train, parts, seed):
    """Each synthetic (or oversampled) row goes to a client with probability proportional to that client's authentic
    sarcastic rows of the same source x dialect group, i.e. clients augment from their own sarcastic seeds."""
    syn = np.flatnonzero(train.origin.values == 'synthetic')
    if len(syn) == 0:
        return parts
    rng = np.random.default_rng(seed + 7919)
    groups, y = train.group.values, train.sarcasm.values
    counts = {}
    for k, p in enumerate(parts):
        for g, n in pd.Series(groups[p][y[p] == 1]).value_counts().items():
            counts.setdefault(g, np.zeros(len(parts)))[k] = n
    extra = [[] for _ in parts]
    for i in syn:
        w = counts.get(groups[i])
        if w is None or w.sum() == 0:
            continue  # no client holds sarcastic seeds of this group (should not occur for SADA)
        extra[rng.choice(len(parts), p=w / w.sum())].append(i)
    return [np.concatenate([p, np.array(e, dtype=int)]) for p, e in zip(parts, extra)]


def describe_clients(train, parts, names):
    rows = []
    for name, p in zip(names, parts):
        f = train.iloc[p]
        rows.append({'client': name, 'n': len(f), 'n_real': int((f.origin == 'real').sum()),
                     'n_synthetic': int((f.origin == 'synthetic').sum()), 'sarcastic_share': float(f.sarcasm.mean()),
                     'sources': ';'.join(f'{s}:{n}' for s, n in f.source_file.value_counts().items())})
    return rows

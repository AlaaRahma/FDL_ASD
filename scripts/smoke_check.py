#!/usr/bin/env python3
"""Import + data + CLI smoke check (no GPU training)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='skip optional heavy imports (torch models)')
    a = ap.parse_args()
    errors = []

    import flasd
    from flasd import data as D
    from flasd.common import AUTHENTIC, SADA_CSV, SADA_TEACHER

    for label, path in [('authentic train', AUTHENTIC / 'train.csv'),
                        ('authentic val', AUTHENTIC / 'validation.csv'),
                        ('authentic test', AUTHENTIC / 'test.csv'),
                        ('SADA', SADA_CSV),
                        ('SADA teacher', SADA_TEACHER)]:
        if not path.exists():
            errors.append(f'missing {label}: {path}')

    auth, val, test = D.load_splits('auth')
    sada, _, _ = D.load_splits('sada')
    sadac, _, _ = D.load_splits('sadac')
    print(f'flasd {flasd.__version__}')
    print(f'auth train={len(auth)} val={len(val)} test={len(test)}')
    print(f'sada train={len(sada)} synthetic={int((sada.origin == "synthetic").sum())}')
    assert len(auth) == 18548, len(auth)
    assert len(val) == 2318 and len(test) == 2318
    assert int((auth.origin == 'synthetic').sum()) == 0
    assert len(sada) == 18548 + 4229
    assert int((sada.origin == 'synthetic').sum()) == 4229
    assert len(sadac) == len(sada)

    from flasd.eval import metrics
    import numpy as np
    m = metrics(np.array([0, 1, 0, 1]), np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]))
    assert 'macro_f1' in m

    if not a.quick:
        import flasd.models  # noqa: F401
        import flasd.train  # noqa: F401
        import flasd.fl  # noqa: F401
        import flasd.classic  # noqa: F401
        import flasd.privacy  # noqa: F401
        import flasd.stats  # noqa: F401
        print('module imports ok')

    # CLI help
    import subprocess
    for cmd in [
        [sys.executable, str(ROOT / 'scripts' / 'run.py'), '--help'],
        [sys.executable, str(ROOT / 'scripts' / 'classic.py'), '--help'],
    ]:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            errors.append(f'CLI failed {" ".join(cmd[1:])}: {r.stderr[-400:]}')
        else:
            print('CLI ok:', Path(cmd[1]).name)

    if errors:
        print('FAIL')
        for e in errors:
            print(' -', e)
        sys.exit(1)
    print('SMOKE OK')


if __name__ == '__main__':
    main()

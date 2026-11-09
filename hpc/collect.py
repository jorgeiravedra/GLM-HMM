#!/usr/bin/env python
"""Stage 3 -- collect fits and run the controls.

Merges work/fits/*.pkl into results/results.pkl plus flat CSVs, then runs the
variance partition and shuffle control (both of which need many refits, so
they belong here rather than in each array task).

    python hpc/collect.py --config hpc/config.json

Outputs
-------
results/loao.csv            one row per (variant, K, holdout): LL, R^2
results/K_selection.csv     paired per-animal gain over K=1, per variant
results/variance_partition_<variant>.csv
results/shuffle_control_<variant>.csv
results/results.pkl         everything, for the notebook

results.pkl holds the full-fit models, posteriors, MAP paths, weight tables
and state diagnostics. Load it with hpc/load_results.py; the SessionSets stay
in work/sessionsets/ and are loaded separately when frame-level alignment is
needed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--skip-controls', action='store_true')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    root = Path(cfg['project_root'])
    sys.path.insert(0, str(root))
    from glmhmm import FitConfig, variance_partition, shuffle_control

    fits_dir = root / 'work' / 'fits'
    res = root / 'results'
    res.mkdir(parents=True, exist_ok=True)

    with open(root / 'work' / 'manifest.json') as f:
        manifest = json.load(f)

    # ------------------------------------------------ gather what exists ---
    files = sorted(fits_dir.glob('*.pkl'))
    print(f'{len(files)} fit files of {len(manifest["tasks"])} tasks',
          flush=True)
    expected = {f'{t["variant"]}_K{t["K"]}_{t["holdout"]}.pkl'
                for t in manifest['tasks']}
    missing = sorted(expected - {f.name for f in files})
    if missing:
        print(f'WARNING: {len(missing)} missing, e.g. {missing[:5]}',
              file=sys.stderr, flush=True)

    loao_rows, full_fits, skipped = [], {}, []
    for fp in files:
        with open(fp, 'rb') as f:
            rec = pickle.load(f)
        if rec['holdout'] == 'full':
            full_fits[(rec['variant'], rec['K'])] = rec
            continue
        if 'skipped' in rec:
            skipped.append((fp.name, rec['skipped']))
            continue
        loao_rows.append(dict(
            variant=rec['variant'], K=rec['K'], animal=rec['holdout'],
            ll_per_frame=rec['ll_per_frame'], n_train=rec['n_train'],
            n_test=rec['n_test'], converged=rec['converged'],
            wall_s=rec['wall_s'],
            **{f'r2_input[{k}]': v
               for k, v in rec.get('r2_input', {}).items()},
            **{f'r2_pred[{k}]': v
               for k, v in rec.get('r2_pred', rec.get('r2', {})).items()}))
    if skipped:
        print(f'{len(skipped)} folds skipped by size guard', flush=True)

    loao = pd.DataFrame(loao_rows)
    if loao.empty:
        print('ERROR: no usable LOAO folds', file=sys.stderr)
        return 1
    loao.to_csv(res / 'loao.csv', index=False)

    # ------------------------------------------------------ K selection ----
    sel = []
    for variant, g in loao.groupby('variant'):
        base = (g.query('K == 1').set_index('animal')['ll_per_frame']
                if (g.K == 1).any() else None)
        for K, gk in g.groupby('K'):
            row = dict(variant=variant, K=int(K),
                       median=gk.ll_per_frame.median(),
                       mean=gk.ll_per_frame.mean(),
                       sd=gk.ll_per_frame.std(), n_animals=len(gk))
            if base is not None:
                d = (gk.set_index('animal')['ll_per_frame'] - base).dropna()
                row.update(delta_median=float(np.median(d)) if len(d) else np.nan,
                           n_improved=int((d > 0).sum()), n_paired=int(len(d)))
            for c in [c for c in gk.columns if c.startswith('r2_')]:
                row[c] = gk[c].median()
            sel.append(row)
    sel = pd.DataFrame(sel).sort_values(['variant', 'K'])
    sel.to_csv(res / 'K_selection.csv', index=False)
    print('\n' + sel.round(4).to_string(index=False), flush=True)

    # --------------------------------------------------------- controls ----
    partitions, shuffles = {}, {}
    if not args.skip_controls:
        an = cfg['analyses']
        for variant in manifest['variants']:
            ssf = root / 'work' / 'sessionsets' / f'{variant}.pkl'
            if not ssf.exists():
                continue
            with open(ssf, 'rb') as f:
                ss = pickle.load(f)
            fc = FitConfig(ridge=cfg['fit']['ridge'],
                           n_restarts=cfg['fit']['n_restarts'],
                           max_iter=cfg['fit']['max_iter'],
                           seed=cfg['fit']['seed'])

            if an.get('variance_partition'):
                t0 = time.time()
                vp = variance_partition(
                    ss, replace(fc, n_states=an['variance_partition_K']),
                    verbose=False)
                vp.to_csv(res / f'variance_partition_{variant}.csv',
                          index=False)
                partitions[variant] = vp
                print(f'[{variant}] partition done in {time.time()-t0:.0f}s',
                      flush=True)

            if an.get('shuffle_control'):
                for K in an['shuffle_K']:
                    t0 = time.time()
                    sh = shuffle_control(ss, replace(fc, n_states=K),
                                         n_shuffle=an['n_shuffle'])
                    sh['variant'] = variant
                    shuffles.setdefault(variant, []).append(sh)
                    print(f'[{variant}] shuffle K={K} in '
                          f'{time.time()-t0:.0f}s', flush=True)
                pd.concat(shuffles[variant]).to_csv(
                    res / f'shuffle_control_{variant}.csv', index=False)

    bundle = dict(
        created=time.strftime('%Y-%m-%d %H:%M:%S'), config=cfg,
        manifest=manifest, loao=loao, K_selection=sel,
        full_fits=full_fits, variance_partition=partitions,
        shuffle={k: pd.concat(v) for k, v in shuffles.items()},
        missing=missing, skipped=skipped)
    with open(res / 'results.pkl', 'wb') as f:
        pickle.dump(bundle, f, protocol=5)
    print(f'\nwrote {res / "results.pkl"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

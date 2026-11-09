#!/usr/bin/env python
"""Stage 1 -- audits, SessionSet construction, task manifest.

Runs once on a single node. Loads the raw dictionaries, runs the alignment
and effective-sample-size audits, builds one SessionSet per design variant,
writes them to work/sessionsets/, and emits work/manifest.json listing every
(variant, K, holdout) fit for the array job to index into.

    python hpc/prepare.py --config hpc/config.json

Audit outputs land in results/ as CSV so they can be read without loading the
pickles. Read them before trusting anything downstream -- in particular,
delta_tail must be clearly positive and clearly beat delta_head, or the
attention gate is on the wrong frames.
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


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--skip-existing', action='store_true',
                    help='leave SessionSets already on disk alone')
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg['project_root'])
    work, results = root / 'work', root / 'results'
    (work / 'sessionsets').mkdir(parents=True, exist_ok=True)
    (work / 'fits').mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root))
    from glmhmm import (DesignConfig, parse_session, audit_lengths,
                        audit_alignment, audit_effective_n, build_sessionset,
                        scaling_stability)
    import glmhmm

    print(f'glmhmm {glmhmm.__version__} | numpy {np.__version__}', flush=True)

    # ---------------------------------------------------------------- data --
    t0 = time.time()
    with open(root / cfg['data']['full_data'], 'rb') as f:
        full_data = pickle.load(f)
    with open(root / cfg['data']['features'], 'rb') as f:
        features = pickle.load(f)
    with open(root / cfg['data']['labels'], 'rb') as f:
        labels = pickle.load(f)
    full_data = {k.strip(): v for k, v in full_data.items()}
    features = {k.strip(): v for k, v in features.items()}
    labels = {k.strip(): v for k, v in labels.items()}
    print(f'loaded {len(full_data)} sessions in {time.time() - t0:.0f}s',
          flush=True)

    group_of_animal = {a: g for g, lst in cfg['animals'].items() for a in lst}
    sf = cfg['session_filter']
    keys = []
    for k in sorted(set(full_data) & set(features)):
        m = parse_session(k, group_of_animal)
        if (m and m['group'] == sf['group']
                and m['day'] not in set(sf['exclude_days'])
                and m['cond_norm'] in set(sf['cond_tokens'])):
            keys.append(k)
    print(f'{len(keys)} sessions pass the filter', flush=True)
    if not keys:
        print('ERROR: no sessions matched the filter', file=sys.stderr)
        return 1

    # -------------------------------------------------------------- audits --
    regions = cfg['regions']
    # Audits only need alignment and sampling settings, which every variant
    # shares; take them from the first one rather than hard-coding a name.
    first = next(iter(cfg['variants'].values()))
    probe = DesignConfig(**{k: v for k, v in first.items()
                            if k != 'restrict_to_label'})

    audit_lengths(keys, full_data, features, labels).to_csv(
        results / 'audit_lengths.csv', index=False)

    al = audit_alignment(keys, full_data, labels,
                         label_align=probe.label_align,
                         max_label_excess=probe.max_label_excess)
    al.to_csv(results / 'audit_alignment.csv', index=False)
    print(f'ALIGNMENT  median delta_tail={al.delta_tail.median():+.3f}  '
          f'delta_head={al.delta_head.median():+.3f}  '
          f'excess-flagged={int(al.excess_flag.sum())}/{len(al)}', flush=True)
    if al.delta_tail.median() <= al.delta_head.median():
        print('WARNING: head alignment matches or beats tail. The assumption '
              'that the streams share their END is wrong; fix before '
              'interpreting any fit.', file=sys.stderr, flush=True)

    audit_effective_n(keys, full_data, features, regions, probe,
                      max_sessions=len(keys)).to_csv(
        results / 'audit_effective_n.csv', index=False)
    scaling_stability(keys, full_data, regions).to_csv(
        results / 'audit_scaling_stability.csv', index=False)

    # -------------------------------------------------------- sessionsets --
    manifest, built = [], {}
    for name, spec in cfg['variants'].items():
        out = work / 'sessionsets' / f'{name}.pkl'
        if args.skip_existing and out.exists():
            with open(out, 'rb') as f:
                ss = pickle.load(f)
            print(f'[{name}] reusing existing SessionSet', flush=True)
        else:
            spec = {k: v for k, v in spec.items() if not k.startswith('_')}
            restrict = spec.pop('restrict_to_label', None)
            dcfg = DesignConfig(**spec)
            print(f'\n[{name}] building (restrict_to_label={restrict})',
                  flush=True)
            ss = build_sessionset(keys, full_data, features, labels, regions,
                                  dcfg, group_of_animal,
                                  restrict_to_label=restrict, verbose=True)
            if ss is None:
                print(f'[{name}] no sessions retained; skipping',
                      file=sys.stderr, flush=True)
                continue
            with open(out, 'wb') as f:
                pickle.dump(ss, f, protocol=5)

        gb = ss.X.nbytes / 1e9
        print(f'[{name}] {ss.X.shape[0]} frames x {ss.X.shape[1]} predictors, '
              f'{ss.n_sessions} sessions, {ss.n_animals} animals, '
              f'X={gb:.2f} GB', flush=True)
        ss.design_report().to_frame('n').to_csv(
            results / f'design_report_{name}.csv')
        ss.summary().to_csv(results / f'sessions_per_animal_{name}.csv')
        built[name] = dict(animals=sorted(map(str, np.unique(ss.animal))),
                           P=int(ss.X.shape[1]), n_frames=int(ss.X.shape[0]),
                           gb=float(gb))

        for K in cfg['fit']['Ks']:
            # 'full' fits every animal, for the reported model. The named
            # holdouts are the LOAO folds that model selection rests on.
            for holdout in ['full'] + built[name]['animals']:
                manifest.append(dict(variant=name, K=int(K), holdout=holdout))

    if not manifest:
        print('ERROR: manifest empty', file=sys.stderr)
        return 1

    with open(work / 'manifest.json', 'w') as f:
        json.dump(dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
                       config=str(args.config), variants=built,
                       tasks=manifest), f, indent=1)

    print(f'\nmanifest: {len(manifest)} tasks across {len(built)} variants')
    print(f'submit with:  sbatch --array=0-{len(manifest) - 1} '
          f'hpc/fit_array.slurm')
    return 0


if __name__ == '__main__':
    sys.exit(main())

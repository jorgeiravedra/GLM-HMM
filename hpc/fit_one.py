#!/usr/bin/env python
"""Stage 2 -- one fit per array task.

Indexes into work/manifest.json by SLURM_ARRAY_TASK_ID and runs a single
(variant, K, holdout) fit, writing work/fits/<variant>_K<K>_<holdout>.pkl.

    python hpc/fit_one.py --config hpc/config.json --task-id 0

holdout == 'full' fits every animal and additionally saves the state
posteriors, MAP path and per-state diagnostics -- that is the model you would
report. A named holdout is one LOAO fold: it fits on the rest and scores the
held-out animal, and those folds are what model selection rests on.

Everything needed downstream is saved: model parameters, held-out metrics,
feature names and groups, and the config that produced them. The SessionSet
is not duplicated into each fit file -- it lives once in work/sessionsets/.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--task-id', type=int,
                    default=int(os.environ.get('SLURM_ARRAY_TASK_ID', -1)))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.task_id < 0:
        print('ERROR: no task id', file=sys.stderr)
        return 2

    with open(args.config) as f:
        cfg = json.load(f)
    root = Path(cfg['project_root'])
    sys.path.insert(0, str(root))

    from glmhmm import FitConfig, SwitchingGLM, SessionSet
    from glmhmm.evaluate import (_subset, state_summary, weight_table,
                                 state_behavior_correspondence,
                                 states_are_variance_regimes)

    with open(root / 'work' / 'manifest.json') as f:
        manifest = json.load(f)
    tasks = manifest['tasks']
    if args.task_id >= len(tasks):
        print(f'ERROR: task {args.task_id} >= {len(tasks)}', file=sys.stderr)
        return 2
    task = tasks[args.task_id]
    variant, K, holdout = task['variant'], task['K'], task['holdout']

    out = root / 'work' / 'fits' / f'{variant}_K{K}_{holdout}.pkl'
    if out.exists() and not args.overwrite:
        print(f'{out.name} exists; skipping (use --overwrite)')
        return 0

    print(f'task {args.task_id}: variant={variant} K={K} holdout={holdout}',
          flush=True)
    print(f'threads: OMP={os.environ.get("OMP_NUM_THREADS")} '
          f'MKL={os.environ.get("MKL_NUM_THREADS")}', flush=True)

    with open(root / 'work' / 'sessionsets' / f'{variant}.pkl', 'rb') as f:
        ss: SessionSet = pickle.load(f)

    fc = FitConfig(n_states=K, ridge=cfg['fit']['ridge'],
                   n_restarts=cfg['fit']['n_restarts'],
                   max_iter=cfg['fit']['max_iter'], tol=cfg['fit']['tol'],
                   sticky=cfg['fit']['sticky'], seed=cfg['fit']['seed'],
                   verbose=True)

    t0 = time.time()
    rec = dict(variant=variant, K=K, holdout=holdout,
               regions=ss.regions, feature_names=ss.feature_names,
               feature_groups=ss.feature_groups,
               n_predictors=int(ss.X.shape[1]),
               fit_config=vars(fc).copy(), task=task,
               glmhmm_version=__import__('glmhmm').__version__)

    if holdout == 'full':
        m = SwitchingGLM(fc).fit(ss, verbose=True)
        rec['model'] = m.to_dict()
        rec['state_summary'] = state_summary(m, ss, fs=40.0)
        rec['weights_by_group'] = weight_table(m, ss, by_group=True)
        rec['weights_full'] = weight_table(m, ss, by_group=False)
        rec['correspondence'] = state_behavior_correspondence(m, ss)
        rec['variance_regimes'] = states_are_variance_regimes(m, ss)
        # Posteriors and MAP path are per-frame; keep them float32 to halve
        # the file. seq_id/animal live in the SessionSet, so the frame order
        # is recoverable without duplicating them here.
        rec['posteriors'] = m.posteriors(ss).astype(np.float32)
        rec['viterbi'] = m.viterbi(ss).astype(np.int8)
        rec['expected_dwell_s'] = m.expected_dwell_s(40.0)
        rec['stationary'] = m.stationary()
        rec['in_sample_ll_per_frame'] = float(m.ll_ / len(ss.X))
    else:
        te_m = ss.animal == holdout
        tr, te = _subset(ss, ~te_m), _subset(ss, te_m)
        P = ss.X.shape[1]
        if len(tr.X) < max(10 * P, 500) or len(te.X) < max(2 * P, 200):
            rec['skipped'] = (f'size guard: train={len(tr.X)} test={len(te.X)} '
                              f'P={P}')
            print('SKIPPED -- ' + rec['skipped'], flush=True)
        else:
            m = SwitchingGLM(fc).fit(tr, verbose=True)
            ss_tot = ((te.Y - te.Y.mean(0)) ** 2).sum(0)
            den = np.where(ss_tot > 0, ss_tot, np.nan)
            # Two R^2 families -- see SwitchingGLM.input_predict.
            #   r2_pred : mixes by p(z_t | y_1..y_{t-1}); a proper forecast but
            #             a function of the signal's own past, so it inflates
            #             with K through level-tracking. Do NOT compare across K.
            #   r2_input: mixes by the stationary distribution, conditioning on
            #             nothing from y. The across-K-comparable encoding
            #             estimate. Identical to r2_pred at K=1.
            r2p = 1.0 - ((te.Y - m.filtered_predict(te)) ** 2).sum(0) / den
            r2i = 1.0 - ((te.Y - m.input_predict(te)) ** 2).sum(0) / den
            rec['model'] = m.to_dict()
            rec['n_train'] = int(len(tr.X))
            rec['n_test'] = int(len(te.X))
            rec['ll_per_frame'] = float(m.log_likelihood(te) / len(te.X))
            rec['r2_pred'] = {r: float(v) for r, v in zip(ss.regions, r2p)}
            rec['r2_input'] = {r: float(v) for r, v in zip(ss.regions, r2i)}
            rec['r2'] = rec['r2_pred']          # back-compat for older readers
            rec['share_weights'] = bool(fc.share_weights)
            rec['converged'] = bool(m.converged_)
            print(f'  ll/frame={rec["ll_per_frame"]:+.4f}\n'
                  f'  r2_input={ {k: round(v, 4) for k, v in rec["r2_input"].items()} }\n'
                  f'  r2_pred ={ {k: round(v, 4) for k, v in rec["r2_pred"].items()} }',
                  flush=True)

    rec['wall_s'] = time.time() - t0
    tmp = out.with_suffix('.pkl.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(rec, f, protocol=5)
    tmp.replace(out)          # atomic: a killed job never leaves a half file
    print(f'wrote {out.name} in {rec["wall_s"]:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

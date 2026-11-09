"""Read HPC output from a mounted project directory, before or after collect.

Works on a partially-complete array: every finished fit is a self-contained
pickle, so the LOAO and K-selection tables can be built from whatever exists.

    import sys; sys.path.insert(0, r'S:\\jmig\\glmhmm')
    from hpc.peek import Peek
    P = Peek(r'S:\\jmig\\glmhmm')

    P.progress()          # how much of the array is done
    P.alignment()         # the gate everything depends on
    P.loao()              # per-fold LL and R^2, from finished fits
    P.K_selection()       # paired per-animal gain over K=1

No glmhmm import is needed for any of the above -- fit records hold plain
arrays, dicts and DataFrames. SessionSets are dataclasses and DO need the
package importable; `sessionset()` says so if it is not.

PANDAS VERSION SKEW. DataFrames pickled by the cluster env may fail to load
under a different local pandas. The tables above are built from scalar fields
only and are immune; `full()` reads the richer records and will say which
fields it could not decode rather than failing outright.
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Scalar/array fields that are safe to read regardless of pandas version.
_SAFE = ('variant', 'K', 'holdout', 'll_per_frame', 'n_train', 'n_test',
         'converged', 'wall_s', 'skipped', 'n_predictors', 'share_weights',
         'glmhmm_version', 'in_sample_ll_per_frame')
# R^2 dicts, expanded into columns. r2_pred is a forecast metric that inflates
# with K; r2_input is the across-K-comparable encoding estimate. 'r2' is the
# legacy key from fits made before the split and maps to r2_pred.
_R2_KEYS = ('r2_input', 'r2_pred', 'r2')


class Peek:
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root)
        if not (self.root / 'work').exists():
            raise FileNotFoundError(
                f'{self.root} has no work/ -- is the drive mounted and is '
                'this the project root (the directory containing glmhmm/, '
                'hpc/, work/, results/)?')
        self._fits: dict[str, dict] | None = None

    # ------------------------------------------------------------ manifest -
    @property
    def manifest(self) -> dict:
        with open(self.root / 'work' / 'manifest.json') as f:
            return json.load(f)

    def progress(self) -> pd.DataFrame:
        """Which tasks have landed. Run this first on an in-flight array."""
        man = self.manifest
        done = {p.stem for p in (self.root / 'work' / 'fits').glob('*.pkl')}
        rows = []
        for t in man['tasks']:
            stem = f"{t['variant']}_K{t['K']}_{t['holdout']}"
            rows.append(dict(**t, done=stem in done))
        df = pd.DataFrame(rows)
        summ = (df.groupby(['variant', 'K'])
                  .done.agg(done='sum', total='size').reset_index())
        n, N = int(df.done.sum()), len(df)
        print(f'{n}/{N} tasks complete ({100 * n / max(N, 1):.0f}%)')
        if n < N:
            miss = df.query('~done')
            print('outstanding:', miss.groupby('variant').size().to_dict())
        return summ

    # -------------------------------------------------------------- audits -
    def audit(self, which: str) -> pd.DataFrame:
        """which: lengths | alignment | effective_n | scaling_stability.
        These are CSVs written by prepare.py, so no version skew."""
        name = {'lengths': 'audit_lengths', 'alignment': 'audit_alignment',
                'effective_n': 'audit_effective_n',
                'scaling_stability': 'audit_scaling_stability'}[which]
        return pd.read_csv(self.root / 'results' / f'{name}.csv')

    def alignment(self) -> pd.DataFrame:
        """Label alignment, with the verdict printed.

        delta_tail must be clearly positive AND clearly beat delta_head. If
        not, the attention gate is on the wrong frames and no fit downstream
        of it means anything.
        """
        al = self.audit('alignment')
        t, h = al.delta_tail.median(), al.delta_head.median()
        print(f'median delta  tail={t:+.3f}  head={h:+.3f}  '
              f'-> {"TAIL (expected)" if t > h else "HEAD -- INVESTIGATE"}')
        print(f'excess-flagged (lost camera frames): '
              f'{int(al.excess_flag.sum())}/{len(al)}')
        neg = al.query('delta_tail <= 0')
        if len(neg):
            print(f'{len(neg)} sessions with non-positive delta_tail: '
                  f'{neg.session.tolist()[:8]}')
        return al

    def design(self, variant: str) -> pd.Series:
        return pd.read_csv(self.root / 'results' / f'design_report_{variant}.csv',
                           index_col=0).squeeze('columns')

    # ---------------------------------------------------------------- fits -
    def _load_fits(self, refresh: bool = False) -> dict[str, dict]:
        if self._fits is not None and not refresh:
            return self._fits
        out = {}
        for p in sorted((self.root / 'work' / 'fits').glob('*.pkl')):
            try:
                with open(p, 'rb') as f:
                    out[p.stem] = pickle.load(f)
            except Exception as e:                      # version skew etc.
                warnings.warn(f'{p.name}: {type(e).__name__}: {e}')
        self._fits = out
        return out

    def loao(self, refresh: bool = False) -> pd.DataFrame:
        """One row per completed cross-validation fold.

        Built from scalar fields only, so it survives pandas version skew
        between the cluster env and the local one.
        """
        rows = []
        for stem, rec in self._load_fits(refresh).items():
            if rec.get('holdout') == 'full' or 'skipped' in rec:
                continue
            if 'll_per_frame' not in rec:
                continue
            d = {k: rec[k] for k in _SAFE if k in rec}
            seen = set()
            for rk in _R2_KEYS:
                if rk not in rec:
                    continue
                label = 'r2_pred' if rk == 'r2' else rk
                if label in seen:
                    continue
                seen.add(label)
                d.update({f'{label}[{k}]': v for k, v in rec[rk].items()})
            d['animal'] = d.pop('holdout')
            rows.append(d)
        if not rows:
            print('no completed folds yet')
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(['variant', 'K', 'animal'])

    def K_selection(self, refresh: bool = False) -> pd.DataFrame:
        """Paired per-animal gain over K=1, per variant.

        delta_median and n_improved are the numbers to read. A median that
        rises while individual animals disagree is not evidence of switching.
        Variants with incomplete folds are marked -- an unbalanced K curve is
        not comparable across K.
        """
        g = self.loao(refresh)
        if g.empty:
            return g
        out = []
        for variant, gv in g.groupby('variant'):
            base = (gv.query('K == 1').set_index('animal')['ll_per_frame']
                    if (gv.K == 1).any() else None)
            n_an = gv.animal.nunique()
            for K, gk in gv.groupby('K'):
                row = dict(variant=variant, K=int(K),
                           median=gk.ll_per_frame.median(),
                           sd=gk.ll_per_frame.std(), n_folds=len(gk),
                           complete=len(gk) == n_an,
                           n_not_converged=int((~gk.converged).sum()))
                if base is not None:
                    d = (gk.set_index('animal')['ll_per_frame'] - base).dropna()
                    row.update(
                        delta_median=float(np.median(d)) if len(d) else np.nan,
                        n_improved=int((d > 0).sum()), n_paired=int(len(d)))
                for c in [c for c in gk.columns if c.startswith('r2_')]:
                    row[c] = gk[c].median()
                out.append(row)
        return pd.DataFrame(out).sort_values(['variant', 'K'])

    def full(self, variant: str, K: int) -> dict:
        """The full-data fit record: model params, posteriors, diagnostics.

        Reports any field that failed to decode rather than raising, so a
        pandas mismatch costs you the DataFrames but not the arrays.
        """
        p = self.root / 'work' / 'fits' / f'{variant}_K{K}_full.pkl'
        if not p.exists():
            raise FileNotFoundError(f'{p.name} not written yet')
        with open(p, 'rb') as f:
            rec = pickle.load(f)
        bad = [k for k, v in rec.items()
               if isinstance(v, pd.DataFrame) and v.empty]
        if bad:
            print(f'empty/undecoded frames: {bad}')
        return rec

    def model(self, variant: str, K: int):
        """Rehydrate a SwitchingGLM. Needs glmhmm importable locally."""
        try:
            from glmhmm import SwitchingGLM
        except ImportError as e:
            raise ImportError(
                'add the project root to sys.path so `glmhmm` imports: '
                f"sys.path.insert(0, r'{self.root}')") from e
        return SwitchingGLM.from_dict(self.full(variant, K)['model'])

    def sessionset(self, variant: str):
        """Load a SessionSet. Needs glmhmm importable AND a compatible
        numpy/pandas; this is the object most exposed to version skew."""
        try:
            import glmhmm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"sys.path.insert(0, r'{self.root}') first") from e
        with open(self.root / 'work' / 'sessionsets' / f'{variant}.pkl', 'rb') as f:
            return pickle.load(f)

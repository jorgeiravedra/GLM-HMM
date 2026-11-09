"""Load HPC results into a notebook.

    import sys; sys.path.insert(0, '/scratch/gpfs/AFALKNER/jmig/glmhmm')
    from hpc.load_results import Results
    R = Results('/scratch/gpfs/AFALKNER/jmig/glmhmm')

    R.K_selection                      # paired gain over K=1, per variant
    R.loao                             # one row per (variant, K, animal)
    R.audit('alignment')               # the pre-fit audits as DataFrames
    m, ss = R.model('ar0', K=3)        # fitted model + its SessionSet
    R.full('ar0', 3)['weights_by_group']

Posteriors and MAP paths in the full fits are in SessionSet frame order, so
`ss.seq_id`, `ss.animal` and `ss.behavior` index them directly.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class Results:
    def __init__(self, project_root: str | Path):
        self.root = Path(project_root)
        with open(self.root / 'results' / 'results.pkl', 'rb') as f:
            self._b = pickle.load(f)
        self._ss_cache: dict[str, object] = {}

    # ------------------------------------------------------------- tables --
    @property
    def loao(self) -> pd.DataFrame:
        return self._b['loao']

    @property
    def K_selection(self) -> pd.DataFrame:
        return self._b['K_selection']

    @property
    def variants(self) -> list[str]:
        return sorted(self._b['manifest']['variants'])

    @property
    def config(self) -> dict:
        return self._b['config']

    def partition(self, variant: str) -> pd.DataFrame:
        return self._b['variance_partition'][variant]

    def shuffle(self, variant: str) -> pd.DataFrame:
        return self._b['shuffle'][variant]

    def audit(self, which: str) -> pd.DataFrame:
        """which: lengths | alignment | effective_n | scaling_stability"""
        name = {'lengths': 'audit_lengths',
                'alignment': 'audit_alignment',
                'effective_n': 'audit_effective_n',
                'scaling_stability': 'audit_scaling_stability'}[which]
        return pd.read_csv(self.root / 'results' / f'{name}.csv')

    # -------------------------------------------------------------- fits ---
    def full(self, variant: str, K: int) -> dict:
        """The full-data fit record: model params, posteriors, diagnostics."""
        return self._b['full_fits'][(variant, int(K))]

    def sessionset(self, variant: str):
        if variant not in self._ss_cache:
            p = self.root / 'work' / 'sessionsets' / f'{variant}.pkl'
            with open(p, 'rb') as f:
                self._ss_cache[variant] = pickle.load(f)
        return self._ss_cache[variant]

    def model(self, variant: str, K: int):
        """Rehydrated SwitchingGLM plus its SessionSet."""
        import sys
        sys.path.insert(0, str(self.root))
        from glmhmm import SwitchingGLM
        rec = self.full(variant, K)
        return SwitchingGLM.from_dict(rec['model']), self.sessionset(variant)

    # ----------------------------------------------------------- summary ---
    def completeness(self) -> pd.DataFrame:
        """What ran and what didn't. Check before interpreting anything:
        a variant missing folds has a biased K-selection curve."""
        rows = []
        for v in self.variants:
            g = self.loao.query('variant == @v')
            rows.append(dict(
                variant=v, n_folds=len(g), n_animals=g.animal.nunique(),
                Ks=sorted(g.K.unique()),
                n_not_converged=int((~g.converged).sum()),
                median_wall_min=round(g.wall_s.median() / 60, 1)))
        out = pd.DataFrame(rows)
        if self._b['missing']:
            print(f'{len(self._b["missing"])} fit files missing')
        if self._b['skipped']:
            print(f'{len(self._b["skipped"])} folds hit the size guard')
        return out

    def alignment_check(self) -> dict:
        """The gate that everything else depends on."""
        al = self.audit('alignment')
        return dict(median_delta_tail=float(al.delta_tail.median()),
                    median_delta_head=float(al.delta_head.median()),
                    n_excess_flagged=int(al.excess_flag.sum()),
                    n_sessions=len(al),
                    tail_wins=bool(al.delta_tail.median()
                                   > al.delta_head.median()))

"""Synthetic data in the lab's schema, and an end-to-end walkthrough.

The generator produces `full_data`, `features` and `sniffing_labels_dict`
dictionaries with the same keys, column names and dtypes as the real
multifiber pickle, with a KNOWN two-state switching structure planted in one
region. Use it to (a) verify the pipeline recovers planted structure, and
(b) calibrate what "no structure" looks like, since the generator can be run
with switch_strength=0.

The synthetic kinematics are generated with a tunable autocorrelation time so
you can see directly how effective sample size, not frame count, governs
whether the design is estimable.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import DesignConfig, FitConfig, BEHAVIOR_GROUPS
from .data import build_sessionset, audit_alignment, audit_effective_n
from .model import SwitchingGLM
from .evaluate import (select_K, fit_loao, state_summary, weight_table,
                       state_behavior_correspondence, shuffle_control,
                       variance_partition)

REGION_IDS = ['PrL (E)', 'PrL (I)', 'vLS (E)', 'vLS (I)', 'POA (E)', 'POA (I)',
              'BNST (E)', 'BNST (I)', 'AH (E)', 'AH (I)', 'MeA (E)', 'MeA (I)',
              'VMH (E)', 'VMH (I)', 'PAG (E)', 'PAG (I)', 'PMv (E)', 'PMv (I)',
              'LHb (E)', 'LHb (I)', 'PA (E)', 'PA (I)', 'NAc (DA)']

FEATURE_NAMES = [
    'intruder centroid roc 100 ms', 'intruder centroid2nose angle',
    'intruder head roc 100 ms', 'intruder tti2head',
    'intruder2resident head-tti', 'intruder2resident head2centroid angle',
    'ious', 'proximity',
    'resident centroid roc 100 ms', 'resident centroid2nose angle',
    'resident head roc 100 ms', 'resident tti2head',
    'resident2intruder head-head', 'resident2intruder head-tti',
    'resident2intruder head2centroid angle',
]


def _ou(T, tau_s, fs, rng):
    """Ornstein-Uhlenbeck trace: sets a controllable autocorrelation time, so
    effective sample size can be dialed independently of frame count."""
    a = np.exp(-1.0 / (tau_s * fs))
    x = np.zeros(T)
    s = np.sqrt(1 - a ** 2)
    for t in range(1, T):
        x[t] = a * x[t - 1] + s * rng.standard_normal()
    return (x - x.mean()) / (x.std() + 1e-12)


def make_synthetic(n_animals: int = 6, sessions_per_animal: int = 3,
                   T: int = 6000, fs: float = 40.0, tau_s: float = 1.5,
                   switch_strength: float = 1.0, noise: float = 1.0,
                   animal_jitter: float = 0.15, seed: int = 20261104):
    """Generate dictionaries matching the real data's schema.

    Planted ground truth
    --------------------
    A two-state process with ~20 s dwell times gates how 'VMH (E)' and
    'PrL (E)' read out the dyad:
        state 0 -> weights RESIDENT (aggressor) kinematics
        state 1 -> weights INTRUDER (subordinate) kinematics
    'NAc (DA)' gets a state-independent kinematic drive (a K=1 signal), and
    the remaining regions get drift plus an attack-event response only. That
    mix is deliberate: a correct pipeline should find K=2 for the first two,
    K=1 for NAc, and no kinematic variance for the rest.

    switch_strength=0 removes the switching entirely -- the null case.
    """
    rng = np.random.default_rng(seed)
    full_data, features, labels = {}, {}, {}
    animals = [f'S{100 + i}' for i in range(n_animals)]
    truth = {}

    # Filters are SHARED across animals, with mild per-animal jitter. This is
    # exactly the assumption a pooled fit makes. If the real filters are
    # animal-specific, a pooled model cannot recover them no matter how much
    # data there is -- worth testing by setting animal_jitter high.
    is_res = np.array([n.startswith('resident') for n in FEATURE_NAMES])
    is_int = np.array([n.startswith('intruder') for n in FEATURE_NAMES])
    W_RES = rng.standard_normal(len(FEATURE_NAMES)) * 0.9
    W_INT = rng.standard_normal(len(FEATURE_NAMES)) * 0.9
    W_RES[~is_res] *= 0.1
    W_INT[~is_int] *= 0.1
    for ai, animal in enumerate(animals):
        w_res = W_RES * (1 + animal_jitter * rng.standard_normal(W_RES.size))
        w_int = W_INT * (1 + animal_jitter * rng.standard_normal(W_INT.size))

        for s in range(sessions_per_animal):
            key = f'{animal}_d{s + 1}_obs_t1'
            F = np.column_stack([_ou(T, tau_s, fs, rng)
                                 for _ in FEATURE_NAMES])

            # Attack bouts: ~8 per session, 1-3 s, matching the real rate.
            atk = np.zeros(T)
            for _ in range(rng.integers(6, 11)):
                o = rng.integers(0, T - int(4 * fs))
                atk[o:o + int(rng.uniform(1, 3) * fs)] = 1.0

            # Observer behavior: a sticky categorical chain, ~43% attention.
            lab = np.zeros(T)
            p = np.array([.06, .07, .05, .08, .43, .10, .05, .08, .08])
            cur = 4
            for t in range(T):
                if rng.random() < 0.004:
                    cur = int(rng.choice(9, p=p / p.sum()))
                lab[t] = cur

            # Latent switching process, ~20 s dwell.
            z = np.zeros(T, int)
            cur_z = 0
            for t in range(T):
                if rng.random() < 1.0 / (30.0 * fs):
                    cur_z = 1 - cur_z
                z[t] = cur_z

            drive_res, drive_int = F @ w_res, F @ w_int
            ev = np.convolve(np.diff(np.concatenate([[0], atk])) > 0,
                             np.exp(-np.arange(int(2 * fs)) / (0.4 * fs)),
                             mode='full')[:T]

            Y = {}
            for r in REGION_IDS:
                drift = _ou(T, 12.0, fs, rng) * 0.35
                if r in ('VMH (E)', 'PrL (E)'):
                    sw = np.where(z == 0, drive_res, drive_int)
                    sig = switch_strength * sw + 0.7 * ev
                elif r == 'NAc (DA)':
                    sig = 0.6 * (drive_res + drive_int) + 0.9 * ev
                else:
                    sig = 0.8 * ev
                y = sig + drift + noise * rng.standard_normal(T)
                Y[r] = (y - y.mean()) / y.std()

            df = pd.DataFrame(Y)
            df['unsupervised labels'] = rng.integers(0, 13, T).astype(float)
            df['attack labels'] = atk
            full_data[key] = df
            features[key] = pd.DataFrame(F, columns=FEATURE_NAMES)
            labels[key] = lab
            truth[key] = z

    group_of_animal = {a: 'OBS' for a in animals}
    return full_data, features, labels, group_of_animal, truth


# ================================================================== walkthrough
def run(n_animals: int = 4, sessions_per_animal: int = 2, T: int = 2000,
        Ks=(1, 2, 3), regions=('VMH (E)', 'PrL (E)', 'NAc (DA)'),
        switch_strength: float = 1.0, verbose: bool = True, seed: int = 3,
        design: DesignConfig | None = None,
        fit: FitConfig | None = None, run_controls: bool = True):
    """End-to-end demo. Returns a dict of every intermediate object.

    Mirrors the order to use on real data: audits -> build -> K selection ->
    fit -> state diagnostics -> controls. Runs in roughly a minute at the
    default sizes; scale n_animals/T up to feel the cost.

    Demo defaults use n_basis=3 and n_ar=0. On real data start from
    DesignConfig() (n_basis=5, n_ar=2) and fit both with and without the AR
    block -- see the note printed in step 3.
    """
    out = {}
    say = print if verbose else (lambda *a, **k: None)

    say('=' * 72)
    say('1. SYNTHETIC DATA (planted: K=2 in VMH (E) and PrL (E); K=1 in NAc)')
    say('=' * 72)
    full_data, features, labels, goa, truth = make_synthetic(
        n_animals=n_animals, sessions_per_animal=sessions_per_animal, T=T,
        switch_strength=switch_strength, seed=seed)
    keys = sorted(full_data)
    say(f'   {len(keys)} sessions, {len(goa)} animals, {T} frames each')
    out.update(full_data=full_data, features=features, labels=labels,
               group_of_animal=goa, truth=truth, keys=keys)

    say('\n' + '=' * 72)
    say('2. AUDITS (on real data these decide whether to proceed at all)')
    say('=' * 72)
    dcfg = DesignConfig(fs=40.0, n_basis=3, n_ar=0) if design is None else design
    fcfg = FitConfig(ridge=5.0, n_restarts=2, max_iter=150) if fit is None else fit
    ne = audit_effective_n(keys, full_data, features, list(regions), dcfg,
                           max_sessions=3)
    say(ne.round(1).to_string(index=False))
    say('   -> compare n_eff_features against the design width below. More '
        'than\n      ~1 predictor per 5 effective samples and ridge shrinks '
        'the filter\n      bank to zero whatever the penalty.')
    out['effective_n'] = ne

    say('\n' + '=' * 72)
    say('3. BUILD POOLED DESIGN')
    say('=' * 72)
    ss = build_sessionset(keys, full_data, features, labels, list(regions),
                          dcfg, goa, verbose=verbose)
    if ss is None:
        raise ValueError(
            f'build_sessionset retained no sessions. With T={T} and a lag '
            f'window of [{dcfg.lag_min_s}, {dcfg.lag_max_s}] s at {dcfg.fs} Hz, '
            f'edge masking removes '
            f'{int((dcfg.lag_max_s - dcfg.lag_min_s) * dcfg.fs)} frames, '
            f'leaving fewer than DesignConfig.min_frames_per_session='
            f'{dcfg.min_frames_per_session}. Raise T or lower '
            'min_frames_per_session.')
    say('\n' + ss.design_report().to_string())
    say('\n   NOTE on n_ar: AR lags of y protect against the HMM segmenting '
        'slow\n   drift into states, but on a smooth signal they also absorb '
        'most of\n   the predictable variance and can swamp the kinematic '
        'filters. Fit\n   both ways on real data and compare; do not accept '
        'either silently.')
    out['sessionset'] = ss

    say('\n' + '=' * 72)
    say('4. SELECT K  (K=1 is ridge regression and is the baseline)')
    say('=' * 72)
    P = ss.X.shape[1]
    n_per_animal = ss.summary().n_frames
    need = max(10 * P, 500)
    short = n_per_animal[n_per_animal.sum() - n_per_animal < need]
    if len(short):
        raise ValueError(
            f'Not enough data for leave-one-animal-out at this design width.\n'
            f'  design width P = {P}; folds need >= {need} training frames.\n'
            f'  frames per animal: {n_per_animal.to_dict()}\n'
            f'Raise n_animals, sessions_per_animal or T, or shrink the design '
            f'(DesignConfig(n_basis=...)).')
    allr, summ = select_K(ss, Ks=Ks, base=fcfg, verbose=False)
    say(summ.round(4).to_string())
    out.update(loao_all=allr, K_summary=summ)

    if not len(summ):
        raise ValueError('select_K returned nothing; see the guards above.')
    key = 'delta_median' if 'delta_median' in summ.columns else 'median'
    best_K = int(summ[key].idxmax())
    say(f'\n   best K by paired held-out LL gain: {best_K}')

    say('\n' + '=' * 72)
    say(f'5. FIT K={best_K} AND INSPECT')
    say('=' * 72)
    m = SwitchingGLM(replace(fcfg, n_states=best_K)).fit(ss)
    say(state_summary(m, ss).round(3).to_string(index=False))
    out.update(model=m, best_K=best_K,
               state_summary=state_summary(m, ss))

    say('\n   per-state RMS weight by feature group:')
    wt = weight_table(m, ss)
    say(wt.round(3).to_string(index=False))
    out['weights'] = wt

    say('\n   state vs observer behavior (ARI > ~0.3 = states are behavior):')
    corr = state_behavior_correspondence(m, ss)
    say(f'   ARI = {corr["adjusted_rand"]:.3f}')
    out['correspondence'] = corr

    if best_K > 1:
        say('\n   recovery of planted state path (synthetic only):')
        z_hat = m.viterbi(ss)
        acc = []
        for si, k in enumerate(ss.session_keys):
            idx = np.flatnonzero(ss.seq_id == si)
            n = min(idx.size, truth[k].size)
            zt = truth[k][:n]
            zh = z_hat[idx][:n]
            a = max((zh == zt).mean(), (zh != zt).mean())  # label switching
            acc.append(a)
        say(f'   median agreement with ground truth = {np.median(acc):.3f} '
            f'(chance 0.5)')
        out['state_recovery'] = float(np.median(acc))

    say('\n' + '=' * 72)
    say('6. CONTROLS')
    say('=' * 72)
    if not run_controls:
        say('   (skipped: run_controls=False)')
        return out
    sh = shuffle_control(ss, replace(fcfg, n_states=best_K),
                         n_shuffle=2, verbose=False)
    if len(sh):
        say(sh.groupby('K').ll_per_frame.median().round(4).to_string())
        say('   -> compare the K-vs-K1 gain here against the real-data gain.')
    out['shuffle'] = sh
    return out

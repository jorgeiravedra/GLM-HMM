"""Cross-validation, model selection, and the diagnostics that decide whether
a fitted model is interpretable."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .config import FitConfig
from .data import SessionSet
from .model import SwitchingGLM


def scaling_stability(keys, full_data, regions, fs: float = 40.0,
                      win_s: float = 30.0) -> pd.DataFrame:
    """Within-session nonstationarity of center and scale, per region.

    Run this whenever dF/F was produced by subtractive baselining rather than
    a fitted airPLS baseline. Unstable scaling matters more for the HMM than
    for the GLM: with Gaussian emissions and a per-state variance, the
    cheapest likelihood gain available to a K>1 model is to segment
    high-noise from low-noise epochs. Those look like states, select K>1, and
    mean nothing about encoding.

    Columns
    -------
    cv_of_sd : SD of the rolling SD / its mean. Stable dF/F sits near
        0.1-0.2; above ~0.4 the gain is wandering within the session.
    sd_ratio : max / min rolling SD. Large values flag epochs where the
        effective scale changed severalfold.
    drift_r  : |corr(rolling mean, time)|. Residual baseline slope that a
        fitted baseline would have removed.
    """
    w = int(win_s * fs)
    rows = []
    for k in keys:
        df = full_data[k]
        for r in regions:
            if r not in df.columns:
                continue
            y = pd.Series(df[r].to_numpy(float))
            rs = y.rolling(w, center=True).std().dropna()
            rm = y.rolling(w, center=True).mean().dropna()
            if len(rs) < 10:
                continue
            t = np.arange(len(rm))
            rows.append(dict(
                session=k, region=r,
                cv_of_sd=float(rs.std() / (rs.mean() + 1e-12)),
                sd_ratio=float(rs.max() / (rs.min() + 1e-12)),
                drift_r=abs(float(np.corrcoef(t, rm)[0, 1]))))
    return pd.DataFrame(rows)


def states_are_variance_regimes(model, ss: SessionSet, fs: float = 40.0,
                                win_s: float = 5.0) -> pd.DataFrame:
    """Do the states differ mainly in local signal variance?

    If so they are scaling artifacts, not encoding regimes. Read alongside
    the sd[...] columns of state_summary: states whose only real difference
    is emission variance are not telling you about behavior.

    `sd_separation` is the spread of per-state median local SD divided by the
    pooled local SD. Above ~0.5 and the states are tracking signal amplitude.
    """
    z = model.viterbi(ss)
    w = int(win_s * fs)
    rows = []
    for i, r in enumerate(ss.regions):
        loc = (pd.Series(ss.Y[:, i]).rolling(w, center=True).std()
                 .bfill().ffill().to_numpy())
        meds = np.array([np.median(loc[z == k]) if (z == k).any() else np.nan
                         for k in range(model.K)])
        d = dict(region=r,
                 sd_separation=float(np.nanstd(meds) / (np.std(loc) + 1e-12)))
        for k in range(model.K):
            d[f'median_local_sd[state{k}]'] = float(meds[k])
        rows.append(d)
    return pd.DataFrame(rows)


def _subset(ss: SessionSet, mask: np.ndarray) -> SessionSet:
    """Subset frames, remapping sequence ids so they stay contiguous."""
    seq = ss.seq_id[mask]
    uniq, remap = np.unique(seq, return_inverse=True)
    return SessionSet(X=ss.X[mask], Y=ss.Y[mask], seq_id=remap,
                      animal=ss.animal[mask],
                      session_keys=[ss.session_keys[i] for i in uniq],
                      feature_names=ss.feature_names,
                      feature_groups=ss.feature_groups,
                      regions=ss.regions, behavior=ss.behavior[mask])


# ================================================================ CV / fit ==
def fit_loao(ss: SessionSet, cfg: FitConfig | None = None,
             verbose: bool = True, min_train: int | None = None,
             min_test: int | None = None) -> pd.DataFrame:
    """Leave-one-animal-out cross-validation.

    The held-out unit is the ANIMAL. Leaving out sessions would let the model
    exploit implant placement, expression level and demonstrator identity --
    all animal-specific -- and inflate every metric. n for any group-level
    claim is animals.

    Returns, per held-out animal:

      ll_per_frame      held-out log-likelihood per frame. THE model-selection
                        criterion: a proper scoring rule for the whole model,
                        state process included.
      r2_pred[<region>]  one-step-ahead R^2 under p(z_t | y_1..y_{t-1}).
                        A proper forecast, but a function of the signal's own
                        past -- with a state-specific bias and a signal
                        autocorrelated over seconds, that posterior acts as an
                        adaptive estimate of the local level of y. This is why
                        predictive R^2 climbs with K whether or not a
                        behavioural code exists. DO NOT compare across K.
      r2_input[<region>] R^2 under the stationary state distribution, which
                        conditions on nothing from y. Measures only what the
                        regressors predict and IS comparable across K.
                        Averages W_k over the stationary distribution, so it
                        is flat across K whether or not state-dependent
                        coding is real: it is the honest encoding estimate,
                        not a test for coding. Use weight_sharing_test for
                        that.

    At K=1 the two are identical by construction.

    min_train / min_test default to multiples of the design width P rather
    than fixed frame counts: a fold with fewer training frames than
    predictors is not a fit, it is interpolation. Raising them is reasonable;
    lowering them below ~2P is not.
    """
    cfg = FitConfig() if cfg is None else cfg
    P = ss.X.shape[1]
    min_train = max(10 * P, 500) if min_train is None else min_train
    min_test = max(2 * P, 200) if min_test is None else min_test
    rows, skipped = [], []
    for a in np.unique(ss.animal):
        te_m = ss.animal == a
        tr, te = _subset(ss, ~te_m), _subset(ss, te_m)
        if len(tr.X) < min_train or len(te.X) < min_test:
            skipped.append((a, len(tr.X), len(te.X)))
            if verbose:
                print(f'  {a}: skipped (train={len(tr.X)} < {min_train} or '
                      f'test={len(te.X)} < {min_test})')
            continue
        m = SwitchingGLM(cfg).fit(tr)
        ll = m.log_likelihood(te) / len(te.X)
        ss_tot = ((te.Y - te.Y.mean(0)) ** 2).sum(0)
        den = np.where(ss_tot > 0, ss_tot, np.nan)
        r2p = 1.0 - ((te.Y - m.filtered_predict(te)) ** 2).sum(0) / den
        r2i = 1.0 - ((te.Y - m.input_predict(te)) ** 2).sum(0) / den
        rows.append(dict(animal=a, K=cfg.n_states, n_test=len(te.X),
                         n_train=len(tr.X), ll_per_frame=float(ll),
                         converged=bool(m.converged_),
                         share_weights=bool(cfg.share_weights),
                         **{f'r2_input[{r}]': float(v)
                            for r, v in zip(ss.regions, r2i)},
                         **{f'r2_pred[{r}]': float(v)
                            for r, v in zip(ss.regions, r2p)}))
        if verbose:
            print(f'  held out {a}: ll/frame={ll:+.4f}  '
                  f'r2_input={dict(zip(ss.regions, np.round(r2i, 4)))}  '
                  f'r2_pred={dict(zip(ss.regions, np.round(r2p, 4)))}')

    if not rows:
        raise ValueError(
            f'fit_loao produced no folds: all {len(skipped)} animals failed '
            f'the size guards (need train >= {min_train}, test >= {min_test} '
            f'frames for P = {P} predictors).\n'
            f'  per-animal (animal, n_train, n_test): {skipped}\n'
            'Either supply more data per animal, shrink the design '
            '(DesignConfig n_basis / drop_integrated_features), or pass '
            'explicit min_train / min_test if you accept the consequences.')
    return pd.DataFrame(rows)


def select_K(ss: SessionSet, Ks=(1, 2, 3, 4), base: FitConfig | None = None,
             verbose: bool = True):
    """Held-out log-likelihood as a function of K. n = animals.

    Uses cross-validated LL rather than AIC/BIC: information criteria assume
    independent samples, and photometry frames are autocorrelated on the order
    of seconds, so the effective sample size is orders of magnitude below the
    frame count and AIC/BIC are badly miscalibrated here.

    K=1 is ridge regression. If the curve does not rise above K=1, there is no
    evidence for switching, and that is the result -- report it rather than
    raising K until something moves.

    Read ll_per_frame and the r2_input columns. The r2_pred columns are a
    forecast metric and inflate with K through the state posterior tracking
    the signal's own level; they are not comparable across K.

    Returns (per_animal_df, summary_df).
    """
    base = FitConfig() if base is None else base
    out = []
    for K in Ks:
        if verbose:
            print(f'K={K}')
        df = fit_loao(ss, replace(base, n_states=K), verbose=verbose)
        if len(df):
            out.append(df)
    if not out:
        raise ValueError('select_K produced no folds for any K; see the '
                         'fit_loao error above for the size guards.')
    allr = pd.concat(out, ignore_index=True)

    summ = (allr.groupby('K')['ll_per_frame']
                .agg(median='median', mean='mean', sd='std', n_animals='size'))
    if 1 in summ.index:
        # Paired per-animal gain over K=1 is the quantity to report: animals
        # differ a lot in absolute LL, so the unpaired difference of medians
        # understates a consistent within-animal improvement.
        base_ll = allr.query('K == 1').set_index('animal')['ll_per_frame']
        gains = {}
        for K in summ.index:
            g = (allr.query('K == @K').set_index('animal')['ll_per_frame']
                 - base_ll).dropna()
            gains[K] = dict(delta_median=float(np.median(g)) if len(g) else np.nan,
                            n_improved=int((g > 0).sum()), n_paired=int(len(g)))
        summ = summ.join(pd.DataFrame(gains).T)
    return allr, summ


# ============================================================ diagnostics ====
def state_behavior_correspondence(model: SwitchingGLM, ss: SessionSet) -> dict:
    """Are the latent states just the observer's own behavior?

    Observer behavior is already a GLM input, so states should capture
    something beyond it. But MOTION ARTIFACT is not in the design: a state
    that is really "the animal is moving" will masquerade as an encoding
    state. High adjusted Rand here is a red flag, not a result.

    Rule of thumb: ARI > ~0.3 means the encoding interpretation is
    unavailable. Refit with restrict_to_label=4 and ask whether states persist
    within attention.
    """
    g = model.posteriors(ss)
    z = g.argmax(1)
    ct = pd.crosstab(pd.Series(z, name='state'),
                     pd.Series(ss.behavior, name='behavior'),
                     normalize='index')
    return dict(crosstab=ct,
                adjusted_rand=float(adjusted_rand_score(ss.behavior, z)),
                max_conditional=float(ct.max(1).mean()),
                state_occupancy=g.mean(0))


def state_summary(model: SwitchingGLM, ss: SessionSet,
                  fs: float = 40.0) -> pd.DataFrame:
    """Occupancy, dwell times, posterior confidence, per-state noise sd.

    Compare expected_dwell_s (from A) against empirical_dwell_s (from the MAP
    path). A large gap means the posterior is flickering and the state
    sequence should not be read as a segmentation. posterior_confidence near
    1/K means states are not separable in the data.
    """
    g = model.posteriors(ss)
    z = g.argmax(1)
    exp_dwell = model.expected_dwell_s(fs)
    rows = []
    for k in range(model.K):
        m = z == k
        d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs = stops - starts
        rows.append(dict(
            state=k, occupancy=float(g[:, k].mean()),
            self_transition=float(model.A[k, k]),
            expected_dwell_s=float(exp_dwell[k]),
            empirical_dwell_s=float(np.median(runs) / fs) if runs.size else np.nan,
            n_bouts=int(runs.size),
            posterior_confidence=float(g[m, k].mean()) if m.any() else np.nan,
            **{f'sd[{r}]': float(np.sqrt(model.S[k][i, i]))
               for i, r in enumerate(ss.regions)}))
    return pd.DataFrame(rows)


def weight_table(model: SwitchingGLM, ss: SessionSet,
                 by_group: bool = True) -> pd.DataFrame:
    """Per-state weights.

    by_group=True gives the RMS weight within each feature group per state --
    the readout for "does this state weight resident vs intruder kinematics
    differently". Norms are scale-dependent on the design, so compare ACROSS
    STATES within a group, never across groups within a state.

    by_group=False gives the full coefficient table for filter reconstruction.
    """
    if not by_group:
        return pd.DataFrame(
            {f'state{k}|{r}': model.W[k, i]
             for k in range(model.K) for i, r in enumerate(ss.regions)},
            index=ss.feature_names)
    rows = []
    for k in range(model.K):
        for i, r in enumerate(ss.regions):
            d = dict(state=k, region=r)
            for grp in np.unique(ss.feature_groups):
                w = model.W[k, i, ss.feature_groups == grp]
                d[grp] = float(np.linalg.norm(w) / np.sqrt(max(w.size, 1)))
            rows.append(d)
    return pd.DataFrame(rows)


# =============================================================== controls ====
def fit_loso_within_animal(ss: SessionSet, cfg: FitConfig | None = None,
                           min_sessions: int = 3,
                           verbose: bool = True) -> pd.DataFrame:
    """Leave-one-session-out, fitting and testing WITHIN each animal.

    The diagnostic for whether complete pooling is the limiting factor. If
    within-animal prediction clearly beats the pooled leave-one-animal-out
    result, the filters are animal-specific and pooling averages them toward
    zero -- a null then reflects heterogeneity, not absent encoding. If the
    two are comparable, pooling is fine and the ceiling is SNR.

    n here is still animals: one row per (animal, held-out session), to be
    aggregated per animal before any test. Each fit sees only one animal's
    data, so it is far more weakly constrained than the pooled fit -- expect
    noisier estimates and read the per-animal spread, not the grand median.
    """
    cfg = FitConfig() if cfg is None else cfg
    P = ss.X.shape[1]
    rows = []
    for a in np.unique(ss.animal):
        am = ss.animal == a
        sess = np.unique(ss.seq_id[am])
        if sess.size < min_sessions:
            if verbose:
                print(f'  {a}: skipped ({sess.size} sessions)')
            continue
        for s_out in sess:
            te_m = am & (ss.seq_id == s_out)
            tr_m = am & (ss.seq_id != s_out)
            tr, te = _subset(ss, tr_m), _subset(ss, te_m)
            if len(tr.X) < max(10 * P, 500) or len(te.X) < max(2 * P, 200):
                continue
            m = SwitchingGLM(cfg).fit(tr)
            ss_tot = ((te.Y - te.Y.mean(0)) ** 2).sum(0)
            den = np.where(ss_tot > 0, ss_tot, np.nan)
            r2i = 1.0 - ((te.Y - m.input_predict(te)) ** 2).sum(0) / den
            rows.append(dict(animal=a, held_out_session=int(s_out),
                             K=cfg.n_states, n_train=len(tr.X),
                             n_test=len(te.X),
                             ll_per_frame=float(m.log_likelihood(te) / len(te.X)),
                             **{f'r2_input[{r}]': float(v)
                                for r, v in zip(ss.regions, r2i)}))
            if verbose:
                print(f'  {a} session {s_out}: '
                      f'r2_input={dict(zip(ss.regions, np.round(r2i, 4)))}')
    return pd.DataFrame(rows)


def weight_sharing_test(ss: SessionSet, Ks=(1, 2, 3), 
                        base: FitConfig | None = None,
                        verbose: bool = True) -> pd.DataFrame:
    """The decisive nested comparison for "do states carry a behavioural code".

    Fits each K twice: unconstrained, and with W pooled across states so only
    the bias and covariance vary. The constrained model keeps the channel
    through which the state posterior tracks the local level of y and removes
    the encoding channel entirely.

    Read `delta_ll_free_minus_shared`: the held-out log-likelihood the
    unconstrained model buys over the shared-weight model at the same K. If
    it is near zero while both beat K=1 by a lot, the K gain is
    level-tracking and the states are not encoding regimes.
    """
    base = FitConfig() if base is None else base
    rows = []
    for K in Ks:
        for sw in (False, True):
            if K == 1 and sw:
                continue                 # identical at K=1
            if verbose:
                print(f'K={K} share_weights={sw}')
            df = fit_loao(ss, replace(base, n_states=K, share_weights=sw),
                          verbose=False)
            if len(df):
                df['K'], df['share_weights'] = K, sw
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    allr = pd.concat(rows, ignore_index=True)

    out = []
    for K in sorted(allr.K.unique()):
        g = allr.query('K == @K')
        free = g.query('~share_weights').set_index('animal')['ll_per_frame']
        shar = g.query('share_weights').set_index('animal')['ll_per_frame']
        d = (free - shar).dropna() if len(shar) else pd.Series(dtype=float)
        row = dict(K=int(K), ll_free=float(free.median()),
                   ll_shared=float(shar.median()) if len(shar) else np.nan,
                   delta_ll_free_minus_shared=float(np.median(d)) if len(d) else np.nan,
                   n_animals=int(len(free)))
        for c in [c for c in g.columns if c.startswith('r2_input[')]:
            row[f'free_{c}'] = g.query('~share_weights')[c].median()
            if len(shar):
                row[f'shared_{c}'] = g.query('share_weights')[c].median()
        out.append(row)
    return pd.DataFrame(out)


def shuffle_control(ss: SessionSet, cfg: FitConfig | None = None,
                    n_shuffle: int = 5, seed: int = 20261104,
                    verbose: bool = False) -> pd.DataFrame:
    """Circularly shift Y within each session, refit, recompute held-out LL.

    The shift preserves each signal's autocorrelation and marginal
    distribution while destroying its alignment to behavior. A K>1 model will
    still find states in shuffled data (drift regimes), so the comparison that
    matters is the K-vs-K=1 GAIN on real data against the same gain on
    shuffles -- not the absolute LL.
    """
    cfg = FitConfig() if cfg is None else cfg
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_shuffle):
        Y = ss.Y.copy()
        for idx, _, _ in ss.sequences():
            n = idx.size
            sh = int(rng.integers(max(n // 10, 1), max(n - n // 10, 2)))
            Y[idx] = np.roll(Y[idx], sh, axis=0)
        ss_s = SessionSet(ss.X, Y, ss.seq_id, ss.animal, ss.session_keys,
                          ss.feature_names, ss.feature_groups, ss.regions,
                          ss.behavior)
        for K in sorted({1, cfg.n_states}):
            c = replace(cfg, n_states=K, n_restarts=max(cfg.n_restarts // 2, 2))
            df = fit_loao(ss_s, c, verbose=verbose)
            if len(df):
                out.append(dict(shuffle=s, K=K,
                                ll_per_frame=float(df.ll_per_frame.median())))
    return pd.DataFrame(out)


def variance_partition(ss: SessionSet, cfg: FitConfig | None = None,
                       groups=('event', 'resident', 'intruder', 'dyadic',
                               'observer', 'sham', 'behavior',
                               'autoregressive'),
                       verbose: bool = False) -> pd.DataFrame:
    """Unique variance per feature group: drop in LOAO R^2 when knocked out.

    Nested-model partitioning, same logic as the banded-ridge analysis in the
    aggression-observation manuscript. Interpretation for the pitch:

      event explains the variance, kinematics add ~0  -> bulk signal encodes
          THAT aggression is occurring, not HOW. A positive, quantitative
          version of the Fig. 3 null, and the cleanest motivation for
          single-unit work.
      kinematics add unique variance                  -> the resident-vs-
          intruder split is live and worth pursuing.
      everything ~0 including NAc (DA)                -> the measurement, not
          the region. Report as a limit of bulk photometry.
    """
    cfg = FitConfig() if cfg is None else cfg
    full = fit_loao(ss, cfg, verbose=verbose)
    if not len(full):
        return pd.DataFrame()
    # Input-only R^2: partitioning the forecast metric would attribute
    # level-tracking to whichever feature group happened to be removed.
    r2cols = [c for c in full.columns if c.startswith('r2_input[')]
    base = full[r2cols].median()
    rows = [dict(model='full', dropped='-', ll=full.ll_per_frame.median(),
                 **base.to_dict())]
    for grp in groups:
        keep = ss.feature_groups != grp
        if keep.all():
            continue
        sub = SessionSet(ss.X[:, keep], ss.Y, ss.seq_id, ss.animal,
                         ss.session_keys,
                         [n for n, k in zip(ss.feature_names, keep) if k],
                         ss.feature_groups[keep], ss.regions, ss.behavior)
        d = fit_loao(sub, cfg, verbose=verbose)
        if not len(d):
            continue
        med = d[r2cols].median()
        rows.append(dict(model=f'drop_{grp}', dropped=grp,
                         ll=d.ll_per_frame.median(), **med.to_dict()))
    out = pd.DataFrame(rows)
    for c in r2cols:
        out[f'unique_{c}'] = np.where(out.model == 'full', np.nan,
                                      base[c] - out[c])
    return out

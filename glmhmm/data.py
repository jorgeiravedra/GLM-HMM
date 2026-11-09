"""Session parsing, pre-fit audits, and pooled design-matrix construction."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (DesignConfig, BEHAVIOR_GROUPS, OTHER_GROUP,
                     BEHAVIOR_REFERENCE, ATTENTION_LABEL)

SESSION_RE = re.compile(
    r'^(?P<animal>[^_]+)_[dD](?P<day>\d+)_(?P<cond>[A-Za-z0-9\-]+)_[tT](?P<trial>\d+)$')


def parse_session(key: str, group_of_animal: dict | None = None) -> dict | None:
    """Parse '1185_d1_obs_t1' -> dict, or None if it does not match.

    Returns `cond` verbatim and `cond_norm` lowercased. Condition tokens are
    NOT normalized onto a canonical set: '1162B_d9_OBSmCD1_t1' and
    '1185_d1_obs_t1' are different paradigms and silently pooling them would
    mix distributions. Audit the token set, then whitelist explicitly.
    """
    m = SESSION_RE.match(key.strip())
    if m is None:
        return None
    d = m.groupdict()
    return dict(session=key.strip(), animal=d['animal'], day=int(d['day']),
                cond=d['cond'], cond_norm=d['cond'].lower(),
                trial=int(d['trial']),
                group=(group_of_animal or {}).get(d['animal'], 'UNKNOWN'))


# =========================================================== pre-fit audits ==
def effective_n(x: np.ndarray, fs: float, max_lag_s: float = 30.0) -> float:
    """n / (1 + 2*sum_k rho_k), summed to the first zero crossing.

    The number that decides whether a design of width P is estimable at all.
    For heavily smoothed kinematics this can be two orders of magnitude below
    the frame count: a 28,000-frame session carried ~22 independent samples in
    the first pass of this dataset.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    x = x - x.mean()
    L = min(int(max_lag_s * fs), x.size - 1)
    denom = float(x @ x)
    if denom <= 0:
        return np.nan
    ac = np.correlate(x, x, 'full')[x.size - 1: x.size - 1 + L] / denom
    zc = np.flatnonzero(ac <= 0)
    ac = ac[1:zc[0]] if zc.size else ac[1:]
    return float(x.size / (1.0 + 2.0 * ac.sum()))


def align_labels(lab: np.ndarray, n: int, how: str = 'tail') -> np.ndarray:
    """Cut a label vector of length >= n down to the neural clock.

    The side-camera label stream starts before intruder entry, so it is
    longer than the neural recording and the two share their END. 'tail'
    returns lab[-n:]. 'head' returns lab[:n] and keeps the pre-intruder
    baseline instead of the real frames -- wrong for this dataset, retained
    only so the two can be compared.
    """
    lab = np.asarray(lab, float)
    if how == 'tail':
        return lab[-n:]
    if how == 'head':
        return lab[:n]
    raise ValueError(f"label_align must be 'tail' or 'head', got {how!r}")


def audit_lengths(keys, full_data, features, labels) -> pd.DataFrame:
    """Neural / feature / label frame counts per session.

    Interpretation of ratio_lab (labels vs min(neural, features)):
      ~1.0 with a small shortfall  -> trailing frames unannotated; truncation
                                      from frame 0 is safe.
      clustered at 0.5 / 0.333     -> labels are at a different rate; upsample,
                                      do not truncate.
      scattered (e.g. 0.64)        -> annotation covers a SUBSET of the
                                      session. Truncating from frame 0 assumes
                                      it starts at the session start. Verify
                                      with audit_alignment before proceeding.
    """
    rows = []
    for k in keys:
        n_neu, n_feat = len(full_data[k]), len(features[k])
        lab = labels.get(k)
        n_lab = int(np.asarray(lab).size) if lab is not None else 0
        common = min([v for v in (n_neu, n_feat, n_lab) if v > 0], default=0)
        rows.append(dict(session=k, n_neu=n_neu, n_feat=n_feat, n_lab=n_lab,
                         ratio_lab=n_lab / max(min(n_neu, n_feat), 1),
                         d_feat=n_feat - n_neu,
                         label_excess=n_lab - min(n_neu, n_feat),
                         common=common))
    return pd.DataFrame(rows)


def audit_alignment(keys, full_data, labels,
                    attention_label: int = ATTENTION_LABEL,
                    attack_col: str = 'attack labels',
                    label_align: str = 'tail',
                    max_label_excess: int = 6000) -> pd.DataFrame:
    """Do the label vector and the neural data share a clock?

    Observers attend during attack bouts, so P(attention | attack) should
    exceed P(attention | no attack). If `delta` is near zero the labels are
    not on the neural clock. This failure is invisible downstream -- every
    fit would silently use a misaligned attention gate -- so run it first.

    `delta_head` repeats the test under head alignment. Tail should win
    clearly; if head wins, the assumption about which end the streams share
    is wrong and nothing else here is trustworthy.

    `label_excess` is n_labels - n_neural. A few thousand is the expected
    pre-intruder baseline. Sessions above max_label_excess are flagged
    (`excess_flag`) as having lost side-camera frames: those cannot be fixed
    by a constant offset and should be dropped, not shifted.
    """
    rows = []
    for k in keys:
        lab = labels.get(k)
        if lab is None or attack_col not in full_data[k].columns:
            continue
        lab = np.asarray(lab, float)
        a = full_data[k][attack_col].to_numpy(float)
        n = min(lab.size, a.size)
        atk = a[:n] > 0.5
        if atk.sum() < 30:
            continue
        d = {}
        for how in ('tail', 'head'):
            att = align_labels(lab, n, how) == attention_label
            d[how] = float(att[atk].mean() - att[~atk].mean())
            if how == label_align:
                p_a, p_n = float(att[atk].mean()), float(att[~atk].mean())
        excess = int(lab.size - a.size)
        rows.append(dict(session=k, n_lab=int(lab.size), n_neu=int(a.size),
                         label_excess=excess,
                         excess_flag=excess > max_label_excess,
                         n_attack_frames=int(atk.sum()),
                         p_att_given_attack=p_a,
                         p_att_given_no_attack=p_n,
                         delta=d[label_align],
                         delta_head=d['head'], delta_tail=d['tail']))
    return pd.DataFrame(rows)


def audit_effective_n(keys, full_data, features, regions, cfg: DesignConfig,
                      max_sessions: int = 10) -> pd.DataFrame:
    """Effective sample size per session, for features and for each signal.

    Compare `n_eff_features` against the design width that build_sessionset
    reports. More than ~1 predictor per 5 effective samples and ridge will
    shrink the filter bank to zero no matter how the penalty is tuned.
    """
    rows = []
    for k in keys[:max_sessions]:
        feat = features[k]
        kin = prune_feature_columns(feat.columns, cfg)
        ne_f = np.nanmedian([effective_n(feat[c].to_numpy(float), cfg.fs)
                             for c in kin])
        d = dict(session=k, n_frames=len(feat), n_eff_features=ne_f)
        for r in regions:
            if r in full_data[k].columns:
                d[f'n_eff[{r}]'] = effective_n(
                    full_data[k][r].to_numpy(float), cfg.fs)
        rows.append(d)
    return pd.DataFrame(rows)


# ============================================================ design pieces ==
def raised_cosine_basis(cfg: DesignConfig):
    """Linearly spaced raised-cosine bumps, 50% overlap, unit-L2 columns.

    Unit-norming makes the ridge penalty comparable across basis functions.
    Returns (B, lags) with lags in SAMPLES; positive lag = feature precedes
    signal (neural response follows behavior).
    """
    lo = int(round(cfg.lag_min_s * cfg.fs))
    hi = int(round(cfg.lag_max_s * cfg.fs))
    lags = np.arange(lo, hi + 1)
    if cfg.n_basis == 1:
        B = np.ones((lags.size, 1))
    else:
        centers = np.linspace(lo, hi, cfg.n_basis)
        width = 2.0 * (centers[1] - centers[0])
        B = np.zeros((lags.size, cfg.n_basis))
        for j, c in enumerate(centers):
            d = (lags - c) / width
            m = np.abs(d) < 1.0
            B[m, j] = 0.5 * (1.0 + np.cos(np.pi * d[m]))
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return B, lags


def _conv(x: np.ndarray, kern: np.ndarray, lag_lo: int) -> np.ndarray:
    """y[t] = sum_l kern[l] x[t - l], l from lag_lo. Edge samples are left in
    place and masked out by _edge_mask."""
    full = np.convolve(x, kern, mode='full')
    return full[np.arange(x.size) - lag_lo]


def _edge_mask(T: int, lags: np.ndarray) -> np.ndarray:
    """Frames where every lag indexes real data (no zero-padding)."""
    m = np.zeros(T, bool)
    lo, hi = int(lags.min()), int(lags.max())
    m[max(hi, 0): T + min(lo, 0)] = True
    return m


def prune_feature_columns(cols, cfg: DesignConfig) -> list[str]:
    out = [c for c in cols if 'label' not in str(c).lower()]
    if cfg.drop_integrated_features:
        out = [c for c in out if 'across lags' not in str(c).lower()]
    return list(out)


def feature_group(col: str) -> str:
    """Assign a kinematic column to resident / intruder / dyadic.

    Order matters: compound 'X2Y' names are relational, not single-animal.
    CAVEAT for interpretation: only the single-animal features can adjudicate
    aggressor- vs subordinate-referenced coding. Symmetric distances
    (head-head), IoU and proximity are dyadic and belong to neither.
    """
    c = str(col).lower().strip()
    if 'intruder2resident' in c or 'resident2intruder' in c:
        return 'dyadic'
    if c.startswith('intruder'):
        return 'intruder'
    if c.startswith('resident'):
        return 'resident'
    return 'dyadic'


def behavior_onehot(labels: np.ndarray, groups: dict | None = None,
                    reference: str = BEHAVIOR_REFERENCE):
    """Fixed-width one-hot over behavior GROUPS, reference level dropped.

    Width is fixed across sessions (groups absent from a session give all-zero
    columns), which is required for pooled fitting. A session-specific
    reference would make weights non-comparable and the design width variable.

    Returns (Z, names, per_frame_group_labels).
    """
    groups = BEHAVIOR_GROUPS if groups is None else groups
    lab = np.asarray(labels, float)
    lut = {}
    for g, ids in groups.items():
        for i in ids:
            lut[int(i)] = g
    assign = np.array([lut.get(int(v), OTHER_GROUP) if np.isfinite(v)
                       else OTHER_GROUP for v in lab], dtype=object)
    names = [g for g in list(groups) + [OTHER_GROUP] if g != reference]
    Z = np.column_stack([(assign == g).astype(float) for g in names])
    return Z, [f'beh_{g}' for g in names], assign


def drift_basis(T: int, cfg: DesignConfig) -> tuple[np.ndarray, list[str]]:
    """Per-session slow raised-cosine basis over session time.

    Absorbs residual baseline wander left by subtractive baselining instead
    of airPLS. Knot spacing (drift_knot_s) must stay far slower than the
    filter window, or the drift block will compete with the kinematic filters
    for fast variance; a warning fires if the margin drops below 8x.

    This is damage control, not a fix. Reprocessing from raw F with airPLS is
    the fix. Report that this block was used and what the knot spacing was.
    """
    n = cfg.n_drift_basis
    if n <= 0:
        return np.zeros((T, 0)), []
    span_s = T / cfg.fs
    window_s = cfg.lag_max_s - cfg.lag_min_s
    eff_knot_s = span_s / max(n - 1, 1)
    if eff_knot_s < 8.0 * window_s:
        warnings.warn(
            f'drift basis knot spacing {eff_knot_s:.1f} s is only '
            f'{eff_knot_s / window_s:.1f}x the {window_s:.1f} s filter window; '
            'the drift block may absorb kinematic variance. Lower '
            'n_drift_basis or shorten the lag window.')
    centers = np.linspace(0, T - 1, n)
    width = 2.0 * (centers[1] - centers[0]) if n > 1 else T
    t = np.arange(T)
    B = np.zeros((T, n))
    for j, c in enumerate(centers):
        d = (t - c) / width
        m = np.abs(d) < 1.0
        B[m, j] = 0.5 * (1.0 + np.cos(np.pi * d[m]))
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return B, [f'drift|b{j}' for j in range(n)]


def _center_scale(A: np.ndarray, how: str) -> tuple[np.ndarray, np.ndarray]:
    """Column center and scale. 'robust' uses median / 1.4826*MAD, which
    stops a single transient setting the scale for a whole session."""
    if how == 'robust':
        c = np.median(A, axis=0)
        s = 1.4826 * np.median(np.abs(A - c), axis=0)
        # MAD is 0 for constant or near-binary columns; fall back to SD.
        bad = s < 1e-9
        if bad.any():
            s = s.copy()
            s[bad] = A[:, bad].std(0)
    elif how == 'zscore':
        c, s = A.mean(0), A.std(0)
    else:
        raise ValueError(f"scaling must be 'zscore' or 'robust', got {how!r}")
    s = np.asarray(s, float).copy()
    s[s < 1e-9] = 1.0
    return np.asarray(c, float), s


def event_regressors(neu: pd.DataFrame, cfg: DesignConfig,
                     col: str = 'attack labels'):
    """Attack onset / offset / sustained, projected onto the temporal basis."""
    a = neu[col].to_numpy(float)
    on = np.zeros_like(a)
    on[1:] = ((a[1:] > 0.5) & (a[:-1] <= 0.5)).astype(float)
    off = np.zeros_like(a)
    off[1:] = ((a[1:] <= 0.5) & (a[:-1] > 0.5)).astype(float)
    sus = (a > 0.5).astype(float)
    B, lags = raised_cosine_basis(cfg)
    lo = int(lags.min())
    cols, names = [], []
    for nm, x in (('attack_onset', on), ('attack_offset', off),
                  ('attack_sustained', sus)):
        for j in range(B.shape[1]):
            cols.append(_conv(x, B[:, j], lo))
            names.append(f'{nm}|b{j}')
    return np.column_stack(cols), names


# =============================================================== SessionSet ==
@dataclass
class SessionSet:
    """Pooled, session-standardized design and signals, kept as sequences.

    X          (T, P)  design matrix
    Y          (T, D)  signals, one column per region
    seq_id     (T,)    session index; HMM sequences never cross sessions
    animal     (T,)    animal id per frame, the unit of cross-validation
    behavior   (T,)    observer behavior group per frame (string)
    """
    X: np.ndarray
    Y: np.ndarray
    seq_id: np.ndarray
    animal: np.ndarray
    session_keys: list
    feature_names: list
    feature_groups: np.ndarray
    regions: list
    behavior: np.ndarray

    def sequences(self):
        """Yield (original_indices, X_seq, Y_seq) per session."""
        for s in range(int(self.seq_id.max()) + 1):
            idx = np.flatnonzero(self.seq_id == s)
            if idx.size:
                yield idx, self.X[idx], self.Y[idx]

    @property
    def n_sessions(self) -> int:
        return len(self.session_keys)

    @property
    def n_animals(self) -> int:
        return int(np.unique(self.animal).size)

    def summary(self) -> pd.DataFrame:
        return (pd.DataFrame(dict(animal=self.animal, seq=self.seq_id))
                  .groupby('animal')
                  .agg(n_frames=('seq', 'size'), n_sessions=('seq', 'nunique')))

    def design_report(self) -> pd.Series:
        return pd.Series({
            'n_frames': len(self.X), 'n_predictors': self.X.shape[1],
            'n_sessions': self.n_sessions, 'n_animals': self.n_animals,
            **pd.Series(self.feature_groups).value_counts().to_dict()})


def build_sessionset(keys, full_data, features, labels, regions,
                     cfg: DesignConfig | None = None,
                     group_of_animal: dict | None = None,
                     restrict_to_label: int | None = None,
                     verbose: bool = True) -> SessionSet | None:
    """Assemble the pooled design across sessions.

    Neural and feature arrays are truncated from the head (they share the
    neural clock); the label vector is cut from the TAIL by default, since
    it starts before intruder entry. See DesignConfig.label_align.

    Scaling and drift are controlled by cfg.scaling and cfg.n_drift_basis;
    see DesignConfig. Both are mitigations for unstable dF/F, not substitutes
    for reprocessing from raw F with airPLS.

    restrict_to_label
        None  -> fit all frames with behavior as a GLM input. Latent states are
                 then "engagement regimes that gate encoding".
        4     -> fit only barrier-attending frames. Latent states are then
                 "distinct encoding regimes within engagement".
        These are different scientific questions. Choose before fitting, not
        after looking at the states.

    All three arrays (neural, features, labels) are truncated to their common
    length. That is only correct if they share frame 0 -- run audit_alignment
    first.
    """
    cfg = DesignConfig() if cfg is None else cfg
    Xs, Ys, seqs, anims, behs, keys_kept = [], [], [], [], [], []
    names = None
    groups = None
    skipped: dict[str, str] = {}

    for k in keys:
        neu, feat = full_data[k], features[k]
        lab_raw = labels.get(k)
        if lab_raw is None:
            skipped[k] = 'no labels'
            continue
        lab_raw = np.asarray(lab_raw, float)

        # The label stream starts before intruder entry, so it runs longer
        # than the neural recording and the two share their END. A large
        # excess instead means side-camera frames were dropped mid-session:
        # no constant offset aligns the streams, so the session is unusable.
        excess = lab_raw.size - min(len(neu), len(feat))
        if excess > cfg.max_label_excess:
            skipped[k] = f'label excess > {cfg.max_label_excess} (lost frames)'
            continue
        if excess < 0:
            skipped[k] = 'labels shorter than neural'
            continue

        T = int(min(len(neu), len(feat), lab_raw.size))
        if T < cfg.min_frames_per_session:
            skipped[k] = 'too short'
            continue
        lab = align_labels(lab_raw, T, cfg.label_align)

        missing = [r for r in regions if r not in neu.columns]
        if missing:
            skipped[k] = 'region column absent'
            continue
        Y = np.column_stack([neu[r].to_numpy(float)[:T] for r in regions])
        bad = np.isnan(Y).mean(0) > 0.01
        if bad.any():
            skipped[k] = 'region not implanted (NaN)'
            continue
        Y = np.where(np.isnan(Y), np.nanmean(Y, axis=0, keepdims=True), Y)

        B, lags = raised_cosine_basis(cfg)
        lo = int(lags.min())

        kin = prune_feature_columns(feat.columns, cfg)
        cols, nm, gp = [], [], []
        for c in kin:
            x = feat[c].to_numpy(float)[:T]
            if np.isnan(x).any():
                i = np.arange(T)
                good = ~np.isnan(x)
                if good.sum() < 0.5 * T:
                    continue  # >50% missing: drop this feature for this session
                x = np.interp(i, i[good], x[good])
            g = feature_group(c)
            for j in range(B.shape[1]):
                cols.append(_conv(x, B[:, j], lo))
                nm.append(f'{c}|b{j}')
                gp.append(g)
        if not cols:
            skipped[k] = 'no usable kinematic features'
            continue
        X = np.column_stack(cols)

        if cfg.include_event and 'attack labels' in neu.columns:
            Xe, nm_e = event_regressors(neu.iloc[:T], cfg)
            X = np.column_stack([X, Xe])
            nm += nm_e
            gp += ['event'] * len(nm_e)

        Zb, nm_b, assign = behavior_onehot(lab)
        if cfg.include_behavior:
            if cfg.behavior_basis:
                cols_b, nm_bx = [], []
                for q, base in enumerate(nm_b):
                    for j in range(B.shape[1]):
                        cols_b.append(_conv(Zb[:, q], B[:, j], lo))
                        nm_bx.append(f'{base}|b{j}')
                Zb_x = np.column_stack(cols_b)
            else:
                Zb_x, nm_bx = Zb, list(nm_b)
            X = np.column_stack([X, Zb_x])
            nm += nm_bx
            gp += ['behavior'] * len(nm_bx)

        # Autoregressive block: one lag set per region, so each signal carries
        # its own history. Without this the HMM segments drift, not encoding.
        for p in range(1, cfg.n_ar + 1):
            for di, r in enumerate(regions):
                yl = np.concatenate([np.full(p, Y[0, di]), Y[:-p, di]])
                X = np.column_stack([X, yl])
                nm.append(f'AR{p}|{r}')
                gp.append('autoregressive')

        mask = _edge_mask(T, lags)
        if restrict_to_label is not None:
            mask &= (lab == restrict_to_label)
        idx = np.flatnonzero(mask)
        if idx.size < cfg.min_frames_per_session:
            skipped[k] = 'too few usable frames after masking'
            continue

        if names is None:
            names, groups = nm, np.asarray(gp)
        elif len(nm) != len(names):
            skipped[k] = f'design width {len(nm)} != {len(names)}'
            continue

        Xi, Yi = X[idx], Y[idx]

        # Residualize against this session's own drift basis. Done per
        # session on the retained frames, so no weight has to generalize
        # across animals -- see DesignConfig.n_drift_basis.
        Zd, _ = drift_basis(idx.size, cfg)
        if Zd.shape[1]:
            Zd = np.column_stack([Zd, np.ones(idx.size)])
            Q, _ = np.linalg.qr(Zd)
            Xi = Xi - Q @ (Q.T @ Xi)
            Yi = Yi - Q @ (Q.T @ Yi)

        if cfg.standardize_within_session:
            cx, sx = _center_scale(Xi, cfg.scaling)
            cy, sy = _center_scale(Yi, cfg.scaling)
            Xi = (Xi - cx) / sx
            Yi = (Yi - cy) / sy

        meta = parse_session(k, group_of_animal)
        animal = meta['animal'] if meta else str(k)
        Xs.append(Xi)
        Ys.append(Yi)
        behs.append(assign[idx])
        seqs.append(np.full(idx.size, len(keys_kept)))
        anims.append(np.full(idx.size, animal, dtype=object))
        keys_kept.append(k)

    if verbose and skipped:
        print(f'  skipped {len(skipped)}/{len(keys)} sessions: '
              f'{pd.Series(list(skipped.values())).value_counts().to_dict()}')
    if not Xs:
        if verbose:
            print('  no usable sessions')
        return None

    ss = SessionSet(X=np.vstack(Xs), Y=np.vstack(Ys),
                    seq_id=np.concatenate(seqs),
                    animal=np.concatenate(anims),
                    session_keys=keys_kept, feature_names=names,
                    feature_groups=groups, regions=list(regions),
                    behavior=np.concatenate(behs))
    if verbose:
        print(f'  built: {len(ss.X)} frames x {ss.X.shape[1]} predictors, '
              f'{ss.n_sessions} sessions, {ss.n_animals} animals')
    return ss

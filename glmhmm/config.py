"""Configuration objects, behavior grouping, and figure defaults."""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- behavior --
# Observer-behavior label vocabulary (sniffing_labels_dict), grouped.
# NOTE: labels 7 and 8 exist in the dictionary but are unassigned here; they
# fall through to OTHER_GROUP. Confirm what they are before interpreting that
# bin. Confirm also that label 4 is the barrier-directed attention class --
# it carries ~43% of frames and is the behaviorally focal one.
BEHAVIOR_GROUPS: dict[str, list[int]] = {
    'Self-directed': [0, 3],      # grooming, scratching
    'Idle':          [5],         # still
    'Exploratory':   [1, 2, 6],   # investigate, rearing, turning
    'Sniffing':      [4],         # barrier-directed attention
}
OTHER_GROUP = 'Other'

# Reference level dropped from the one-hot, so it is not collinear with the
# per-state bias. Sniffing/attention is the modal and focal class, so every
# other group's weight reads as a contrast against attending.
BEHAVIOR_REFERENCE = 'Sniffing'

ATTENTION_LABEL = 4


@dataclass(frozen=True)
class DesignConfig:
    """Design-matrix construction.

    Parameters
    ----------
    fs
        Per-channel photometry rate (Hz). VERIFY against the DAQ: every lag in
        the model is expressed in seconds through this number.
    lag_min_s, lag_max_s
        Filter support. Negative = acausal (neural leads behavior).
    n_basis
        Raised-cosine bumps spanning the lag window. Keep small. The feature
        autocorrelation time, not the frame rate, bounds how finely a filter
        can be resolved; 5-6 bumps over ~3.5 s is already generous.
    n_ar
        Autoregressive lags of y included as regressors. Without these the HMM
        will segment slow photometry drift into "high" and "low" states that
        have nothing to do with encoding. Default 2. Set 0 only deliberately.
    drop_integrated_features
        Drop '* across lags' columns. They are lag-free summaries built for a
        model without temporal filters; the basis expansion recovers that
        integration itself, so keeping them roughly doubles the design width
        for no new information.
    include_behavior
        Add the one-hot observer-behavior block as a GLM input.
    behavior_basis
        Lag-expand the behavior one-hot onto the temporal basis. Costs
        n_basis x more columns; usually unnecessary.
    include_event
        Add attack onset / offset / sustained regressors. These are the
        control that matters: bulk PrL is strongly driven at attack onset but
        does not discriminate attack vigor, so an event-only model that
        explains the variance while kinematics add nothing IS a result.
    standardize_within_session
        Z-score X and y within session before pooling. Equivalent to session
        fixed effects for mean and scale. Assumes filters are stationary
        across sessions -- the central assumption of pooling, state it.
    scaling
        'zscore' (mean / SD) or 'robust' (median / 1.4826 * MAD). Use
        'robust' when dF/F scaling is unstable: one bad transient will
        otherwise set the scale for a whole session. Note this fixes outlier
        sensitivity only, NOT time-varying gain -- that is what the drift
        basis is for.
    n_drift_basis
        Per-session slow raised-cosine time basis used to remove residual
        baseline wander left by subtractive baselining instead of airPLS.
        Set 0 to disable.

        X and Y are RESIDUALIZED against this basis within each session; the
        basis is not added as model columns. An earlier version added it as a
        shared-weight feature group, which cannot work: drift regressors are
        per-session time ramps, so one weight per column fitted across
        sessions is meaningless on a held-out animal, and leave-one-animal-out
        log-likelihood diverged. Residualizing removes the wander without any
        weight that has to generalize.

        This is damage control, not a substitute for reprocessing from raw F
        with airPLS. It is visible and reportable, which rolling z-scoring is
        not: rolling normalization acts as a signal-dependent adaptive
        high-pass, removes the slow task-related modulation of interest, and
        makes the resulting R^2 uninterpretable. Do not use it.
    drift_knot_s
        Spacing of drift basis centers, in seconds. Must stay far slower than
        the filter window (lag_max_s - lag_min_s) so the drift block cannot
        compete with the kinematic filters for fast variance. The default of
        45 s is ~13x the stock 3.5 s window. A warning is issued below ~8x.
    label_align
        How the observer-label vector is aligned to the neural clock.
        'tail' (default) keeps the LAST n labels, where n is the neural
        length: the side-camera label vector starts before intruder entry, so
        the two streams share their END, not their start. 'head' keeps the
        first n and is almost certainly wrong for this dataset -- it retains
        the pre-intruder frames and discards real ones, which silently
        misaligns the attention gate by the length of the baseline period.
    max_label_excess
        Sessions where (n_labels - n_neural) exceeds this are dropped. A
        normal pre-intruder baseline runs a few thousand frames; a much
        larger excess means side-camera frames were lost mid-session, so the
        two streams cannot be aligned by a constant offset at all and the
        session is unusable rather than merely shifted.
    min_frames_per_session
        Sessions with fewer usable frames are dropped (and reported).
    """
    fs: float = 40.0
    lag_min_s: float = -1.5
    lag_max_s: float = 2.0
    n_basis: int = 5
    n_ar: int = 2
    drop_integrated_features: bool = True
    include_behavior: bool = True
    behavior_basis: bool = False
    include_event: bool = True
    standardize_within_session: bool = True
    label_align: str = 'tail'        # 'tail' | 'head'
    max_label_excess: int = 6000
    scaling: str = 'zscore'          # 'zscore' | 'robust'
    n_drift_basis: int = 0           # 0 disables; try 8-15 for a 9-min session
    drift_knot_s: float = 45.0
    min_frames_per_session: int = 400


@dataclass(frozen=True)
class FitConfig:
    """EM and regularization.

    Parameters
    ----------
    n_states
        K. K=1 is exactly ridge regression and is the baseline the switching
        model must beat on held-out log-likelihood to be worth reporting.
    ridge
        L2 penalty on state weights in the M-step. The bias is unpenalized.
    n_restarts
        EM is non-convex. Use >=5 for anything you intend to report.
    sticky
        Dirichlet pseudo-count added to self-transitions. Raise it if states
        flicker at the frame scale; dwell times below ~1 s are not credible
        given indicator kinetics.
    var_floor
        Lower bound on emission variance, guards degenerate states.
    share_weights
        Constrain every state to the SAME regression weights W, leaving only
        the bias b_k and covariance Sigma_k free. This is the decisive nested
        model for whether states carry a behavioural code: it keeps the
        level/variance channel (through which the state posterior tracks the
        local level of y) and removes the encoding channel entirely. If a
        share_weights fit recovers most of an unconstrained model's held-out
        gain over K=1, the states are not encoding regimes.
    hier_tau
        Partial pooling of regression weights across animals. None (default)
        fits one shared W per state -- complete pooling, which assumes the
        filters are stationary across animals. A float instead fits
        W_a = W_group + delta_a per animal, penalising each deviation by
        hier_tau: large tau approaches complete pooling, small tau approaches
        a separate fit per animal.

        Motivation: fibre placement within a region, expression level and the
        observer's platform position all vary between animals. Complete
        pooling averages animal-specific filters toward zero, so a null can
        reflect heterogeneity rather than absent encoding.

        Under leave-one-animal-out the held-out animal has no delta_a, so
        prediction uses W_group alone. The benefit is therefore a better-
        estimated group filter, not a personalised one, and the gain is
        expected to be modest. To ask whether pooling is the limiting factor,
        use evaluate.fit_loso_within_animal instead, which keeps the fit
        inside one animal.
    seed
        Restart seeding. Fixed by default for reproducibility.
    """
    n_states: int = 3
    ridge: float = 10.0
    n_restarts: int = 5
    max_iter: int = 200
    tol: float = 1e-5
    var_floor: float = 1e-4
    sticky: float = 0.0
    share_weights: bool = False
    hier_tau: float | None = None
    seed: int = 20261104
    verbose: bool = False


# ----------------------------------------------------------------- figures --
# Okabe-Ito: colorblind-safe qualitative palette.
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
             '#E69F00', '#56B4E9', '#F0E442', '#000000']


def set_nature_rc(scale: float = 1.0) -> None:
    """Matplotlib defaults approximating Nature figure specs.

    7 pt body text, 5-7 pt tick labels, 0.5 pt rules, no top/right spines,
    TrueType fonts embedded so the PDF/SVG stays editable in Illustrator.
    Single-column width is 89 mm, double is 183 mm.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7 * scale,
        'axes.labelsize': 7 * scale,
        'axes.titlesize': 7 * scale,
        'xtick.labelsize': 6 * scale,
        'ytick.labelsize': 6 * scale,
        'legend.fontsize': 6 * scale,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.0,
        'ytick.major.size': 2.0,
        'lines.linewidth': 0.9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
    })


MM = 1.0 / 25.4  # mm -> inches

# glmhmm

Switching linear-Gaussian regression (input-driven HMM) for multi-fiber
photometry. Falkner lab / observational aggression.

```
z_t | z_{t-1} ~ Categorical(A[z_{t-1}, :])
y_t | z_t = k  ~ N(W_k x_t + b_k, S_k)
```

`x_t` stacks temporally-filtered dyad kinematics, attack-event regressors,
one-hot observer-behavior group, and optional autoregressive lags of `y`.

**Naming.** This is not the Ashwood/Pillow GLM-HMM, which has
Bernoulli/categorical emissions over discrete choices. Emissions here are
continuous Gaussian. In methods text, call it a switching linear-Gaussian
regression, or an input-driven HMM with Gaussian emissions.

## Install

Drop the `glmhmm/` folder next to your notebook, or:

```bash
conda activate postdoc-env
pip install -e .
```

Requires numpy, scipy, pandas, scikit-learn, matplotlib.

## Verify the install

```python
import glmhmm
out = glmhmm.demo.run()          # ~2 min
```

Generates synthetic sessions in the lab's schema with a known two-state
structure planted in `VMH (E)` and `PrL (E)` (state 0 weights resident
kinematics, state 1 weights intruder), a state-independent kinematic drive in
`NAc (DA)`, and event-only responses elsewhere. A correct pipeline selects
K=2, recovers the resident/intruder split in the weight table, returns ARI
near 0 against observer behavior, and shows the K=2 advantage vanishing under
circular shift.

`make_synthetic(switch_strength=0)` gives the null case — run it to see what
"no structure" looks like before interpreting real output.

## Sizing

Cross-validation folds need enough frames to estimate the design. `fit_loao`
requires `10*P` training and `2*P` test frames by default, where `P` is the
design width (`ss.design_report()`). With the stock `DesignConfig` and ~15
kinematic features, `P` is around 100, so plan on >=1000 training frames per
fold — i.e. at least 3-4 animals with a few thousand usable frames each.
Undersized runs raise with the per-animal counts rather than failing
downstream.

Edge masking removes `(lag_max_s - lag_min_s) * fs` frames per session (140
at the defaults), and `DesignConfig.min_frames_per_session` drops what is
left if it is too short.

## Label alignment

The observer-label vector is produced from the side camera and starts before
intruder entry, so it is longer than the neural recording and the two streams
share their **end**. `build_sessionset` cuts labels from the tail
(`DesignConfig.label_align='tail'`); neural and feature arrays are cut from
the head, since they share the neural clock.

```python
al = audit_alignment(keys, full_data, sniffing_labels_dict)
display(al[['session', 'label_excess', 'excess_flag',
            'delta_tail', 'delta_head']].round(3))
print(al.delta_tail.median(), al.delta_head.median())
```

`delta_tail` should be clearly positive and clearly beat `delta_head`.
Observers attend during attack bouts, so P(attention | attack) exceeds
P(attention | no attack) only when the gate is on the right frames. If head
alignment wins, the assumption about which end the streams share is wrong.

`label_excess` is `n_labels - n_neural` — the pre-intruder baseline, normally
a few thousand frames. Above `max_label_excess` (default 6000), side-camera
frames were lost mid-session: no constant offset aligns the streams, so those
sessions are dropped rather than shifted. They appear in the
`build_sessionset` skip report.

## Unstable dF/F

If baselines were produced by subtraction rather than a fitted airPLS
baseline, run `scaling_stability` before fitting. This matters more for the
HMM than the GLM: with Gaussian emissions and a per-state variance, the
cheapest likelihood gain available to a K>1 model is segmenting high-noise
from low-noise epochs. Those look like states and mean nothing.

```python
st = scaling_stability(keys, full_data, REGIONS)
st.groupby('region')[['cv_of_sd', 'sd_ratio', 'drift_r']].describe()
```

`cv_of_sd` near 0.1-0.2 is stable; above ~0.4 the gain wanders within the
session. High `drift_r` is residual baseline slope.

Mitigations, neither a substitute for reprocessing from raw F:

```python
dcfg = DesignConfig(n_basis=5, n_ar=2,
                    scaling='robust',      # median / MAD, not mean / SD
                    n_drift_basis=10,      # slow per-session time basis
                    drift_knot_s=45.0)
```

`scaling='robust'` stops one transient setting a session's scale; it does not
address time-varying gain. `n_drift_basis` adds a slow raised-cosine time
basis as a `'drift'` nuisance group, absorbing baseline wander without
altering the signal. Keep knot spacing well above the filter window — a
warning fires below 8x. `variance_partition` reports the drift group's
contribution like any other.

Do **not** rolling z-score. It acts as a signal-dependent adaptive high-pass,
removes the slow task-related modulation of interest, and makes the resulting
R² uninterpretable.

Then check the states aren't amplitude regimes:

```python
display(states_are_variance_regimes(m, ss))
```

Read `sd_separation` next to the `sd[...]` columns of `state_summary`, and
compare it against a fit on uncorrupted data if you have one — it is a
comparison, not a threshold, since real states can also differ in amplitude.

## Real-data workflow

```python
from glmhmm import *

GROUP_OF_ANIMAL = ({a: 'EXP' for a in agg_animals}
                   | {a: 'OBS' for a in obs_animals}
                   | {a: 'NON' for a in non_animals})

keys = [k for k in sorted(set(full_data) & set(features))
        if (m := parse_session(k, GROUP_OF_ANIMAL))
        and m['group'] == 'OBS' and m['day'] != 9 and m['cond_norm'] == 'obs']

# 1. Audits. Do not skip the alignment one.
display(audit_lengths(keys, full_data, features, sniffing_labels_dict))
al = audit_alignment(keys, full_data, sniffing_labels_dict)
print(al.delta.describe())        # must be clearly positive

dcfg = DesignConfig(fs=40.0, n_basis=5, n_ar=2)
display(audit_effective_n(keys, full_data, features, REGIONS, dcfg))

# 2. Build
REGIONS = ['VMH (E)', 'VMH (I)', 'PrL (E)', 'PrL (I)', 'NAc (DA)']
ss = build_sessionset(keys, full_data, features, sniffing_labels_dict,
                      REGIONS, dcfg, GROUP_OF_ANIMAL)
print(ss.design_report())

# 3. Select K. K=1 is ridge regression and is the baseline.
allr, summ = select_K(ss, Ks=(1, 2, 3, 4))

# 4. Fit and inspect
m = SwitchingGLM(FitConfig(n_states=3)).fit(ss)
display(state_summary(m, ss))
corr = state_behavior_correspondence(m, ss)
display(weight_table(m, ss))

# 5. Controls
display(shuffle_control(ss, FitConfig(n_states=3)))
display(variance_partition(ss, FitConfig(n_states=3)))

# 6. Figures
set_nature_rc()
fig, _ = plot_K_selection(allr, summ)
savefig(fig, 'fig/K_selection')
```

## The checks that decide whether output is interpretable

| Check | Where | Fails if |
|---|---|---|
| Label clock shared with neural clock | `audit_alignment` | `delta` near zero — labels offset or decimated; truncation is wrong |
| Design estimable at all | `audit_effective_n` vs `design_report` | more than ~1 predictor per 5 effective samples |
| Switching beats ridge | `select_K` | `delta_median` ≤ 0 at every K > 1 |
| States are not behavior | `state_behavior_correspondence` | ARI > ~0.3 |
| States are a real segmentation | `state_summary` | `posterior_confidence` near 1/K, or `empirical_dwell_s` far below `expected_dwell_s` |
| Gain is not drift | `shuffle_control` | shuffled K-vs-K1 gain matches real |

## Design decisions to make before fitting, not after

**What the states mean.** `restrict_to_label=None` fits all frames with
behavior as an input, so states are engagement regimes that gate encoding.
`restrict_to_label=4` fits only barrier-attending frames, so states are
encoding regimes within engagement. Different questions. Pick one.

**The AR block.** `n_ar>0` stops the HMM segmenting slow photometry drift
into "high" and "low" states. But on a smooth signal, AR lags also absorb
most of the predictable variance and can swamp the kinematic filters — this
is visible in the demo. Fit both ways and compare; do not accept either
silently.

**Pooling assumes stationary filters.** Within-session standardization gives
session fixed effects for mean and scale, but a single `W` per state is
shared across animals. If the real filters are animal-specific, no amount of
data recovers them. `make_synthetic(animal_jitter=...)` lets you see how fast
recovery degrades.

**Feature groups.** Only the single-animal features (`resident *`,
`intruder *`) can adjudicate aggressor- vs subordinate-referenced coding.
Symmetric distances, IoU and proximity are dyadic and belong to neither.
`weight_table` norms are comparable across states within a group, never
across groups within a state.

**n is animals.** `fit_loao` holds out animals, not sessions. Sessions within
an animal share implant placement, expression level and demonstrator;
treating them as independent is pseudoreplication.

## Reading `variance_partition`

| Pattern | Reading |
|---|---|
| `event` carries the variance, kinematics add ~0 | bulk signal encodes *that* aggression is occurring, not *how* — a positive, quantitative version of the Fig. 3 null |
| kinematics add unique variance | the resident-vs-intruder split is live |
| everything ~0, including `NAc (DA)` | a limit of bulk photometry at this effective sample size, not a result about any region |

## Notes

- Seeded throughout (`FitConfig.seed`); EM restarts are deterministic.
- States are canonically ordered by occupancy after fitting, so weight tables
  are comparable across restarts, folds and regions.
- Held-out R² uses `filtered_predict` (causal state beliefs). Smoothed
  posteriors see the future; an R² built from them is circular.
- `select_K` uses cross-validated log-likelihood, not AIC/BIC: information
  criteria assume independent samples, and photometry frames are
  autocorrelated on the order of seconds.

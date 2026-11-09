"""Publication-spec figures. Vector output, colorblind-safe, 7 pt type.

All functions return (fig, ax) and draw nothing to disk; call savefig() to
write PDF + PNG. Panel letters are added by the caller or by savefig's
`panel` argument so multi-panel assembly stays explicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import OKABE_ITO, set_nature_rc, MM


def savefig(fig, path_stem: str, formats=('pdf', 'png')) -> list[str]:
    """Write vector + raster copies. PDF keeps text as editable TrueType."""
    out = []
    for f in formats:
        p = f'{path_stem}.{f}'
        fig.savefig(p, format=f)
        out.append(p)
    return out


def _panel(ax, letter: str | None):
    if letter:
        ax.text(-0.18, 1.08, letter, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='left')


def plot_K_selection(allr: pd.DataFrame, summ: pd.DataFrame,
                     shuffle: pd.DataFrame | None = None,
                     panel: str | None = 'a', ax=None):
    """Held-out LL per frame vs K, per-animal lines over the median.

    Per-animal lines are the point: n is animals, and a consistent within-
    animal gain is the evidence for switching. A median that rises while
    individual animals disagree is not.
    """
    set_nature_rc()
    if ax is None:
        fig, ax = plt.subplots(figsize=(45 * MM, 42 * MM))
    else:
        fig = ax.figure
    for i, (a, g) in enumerate(allr.groupby('animal')):
        g = g.sort_values('K')
        ax.plot(g.K, g.ll_per_frame, '-', lw=0.5, alpha=0.45,
                color=OKABE_ITO[i % len(OKABE_ITO)], zorder=1)
    med = allr.groupby('K').ll_per_frame.median()
    ax.plot(med.index, med.values, 'o-', color='k', ms=3, lw=1.1, zorder=3,
            label='median')
    if shuffle is not None and len(shuffle):
        sm = shuffle.groupby('K').ll_per_frame.median()
        ax.plot(sm.index, sm.values, 's--', color='0.55', ms=2.5, lw=0.8,
                zorder=2, label='circular shift')
    ax.set_xlabel('number of states, $K$')
    ax.set_ylabel('held-out log-likelihood\nper frame')
    ax.set_xticks(sorted(allr.K.unique()))
    ax.legend(loc='best')
    _panel(ax, panel)
    fig.tight_layout()
    return fig, ax


def plot_loao_r2(allr: pd.DataFrame, K: int, regions: list,
                 panel: str | None = 'b', ax=None):
    """Held-out filtered-prediction R^2 per region, one point per animal."""
    set_nature_rc()
    d = allr.query('K == @K')
    cols = [f'r2[{r}]' for r in regions if f'r2[{r}]' in d.columns]
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(40, 14 * len(cols)) * MM, 42 * MM))
    else:
        fig = ax.figure
    rng = np.random.default_rng(0)
    for i, c in enumerate(cols):
        v = d[c].dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=6,
                   color=OKABE_ITO[i % len(OKABE_ITO)], alpha=0.85,
                   linewidths=0, zorder=2)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color='k', lw=1.0, zorder=3)
    ax.axhline(0, color='0.6', lw=0.5, ls=':', zorder=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[3:-1] for c in cols], rotation=45, ha='right')
    ax.set_ylabel('held-out $R^2$')
    ax.set_xlim(-0.6, len(cols) - 0.4)
    _panel(ax, panel)
    fig.tight_layout()
    return fig, ax


def plot_state_raster(model, ss, session_index: int = 0, fs: float = 40.0,
                      region_index: int = 0, panel: str | None = 'c',
                      max_s: float = 120.0):
    """State posteriors over one session, with the signal above.

    Read the posterior band first: if it is grey rather than saturated, the
    states are not separable and the MAP path below it is not a segmentation.
    """
    set_nature_rc()
    idx = np.flatnonzero(ss.seq_id == session_index)
    n = min(idx.size, int(max_s * fs))
    idx = idx[:n]
    g = model.posteriors(ss)[idx]
    t = np.arange(n) / fs
    fig, axes = plt.subplots(2, 1, figsize=(120 * MM, 48 * MM), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1], hspace=0.15))
    axes[0].plot(t, ss.Y[idx, region_index], color='k', lw=0.6)
    axes[0].set_ylabel(f'{ss.regions[region_index]}\n(z)')
    for k in range(model.K):
        axes[1].fill_between(t, 0, g[:, k], step='mid', alpha=0.75,
                             color=OKABE_ITO[k % len(OKABE_ITO)],
                             linewidth=0, label=f'state {k}')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('posterior')
    axes[1].set_xlabel('time (s)')
    axes[1].legend(ncol=model.K, loc='upper center',
                   bbox_to_anchor=(0.5, -0.45))
    axes[0].set_title(ss.session_keys[session_index], fontsize=6, loc='left')
    _panel(axes[0], panel)
    fig.subplots_adjust(bottom=0.32)
    return fig, axes


def plot_weight_matrix(wt: pd.DataFrame, region: str, panel: str | None = 'd',
                       ax=None):
    """Per-state RMS weight by feature group, one region.

    Compare across states within a column. Comparing across columns within a
    state is not meaningful -- group norms depend on how many columns each
    group contributes.
    """
    set_nature_rc()
    d = wt.query('region == @region').set_index('state')
    cols = [c for c in d.columns if c != 'region']
    M = d[cols].to_numpy(float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(42, 13 * len(cols)) * MM,
                                        (16 + 9 * len(d)) * MM))
    else:
        fig = ax.figure
    im = ax.imshow(M, cmap='cividis', aspect='auto')   # perceptually uniform
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f'state {i}' for i in d.index])
    ax.set_title(region, fontsize=7, loc='left')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    fontsize=5,
                    color='white' if M[i, j] < M.max() * 0.6 else 'black')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('RMS weight', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    _panel(ax, panel)
    fig.tight_layout()
    return fig, ax


def plot_state_behavior(corr: dict, panel: str | None = 'e', ax=None):
    """P(behavior | state). High concentration on one behavior per state means
    the states are tracking observer behavior or its motion artifact, not
    encoding regime -- the ARI in the title is the number to read."""
    set_nature_rc()
    ct = corr['crosstab']
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(45, 15 * ct.shape[1]) * MM,
                                        (18 + 9 * ct.shape[0]) * MM))
    else:
        fig = ax.figure
    im = ax.imshow(ct.to_numpy(), cmap='cividis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns, rotation=45, ha='right')
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels([f'state {i}' for i in ct.index])
    ax.set_title(f"ARI = {corr['adjusted_rand']:.3f}", fontsize=7, loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('P(behavior | state)', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    _panel(ax, panel)
    fig.tight_layout()
    return fig, ax

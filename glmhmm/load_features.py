#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load curated per-session feature tables into the `features` dictionary.

Replaces the feats_list1..5 / set-union pattern. That version produced the
union of five overlapping hand-lists, which made the final column set an
accident of which lists happened to differ; here the set is stated once with
provenance, and the union is only used to reconstruct the legacy set for
comparison.

Key differences from the previous script:
  - one explicit curated list, grouped by what each feature measures
  - observer-referenced features and their sham controls included
  - every session validated against the same schema before being accepted,
    because pooling in glmhmm requires a constant feature set and a session
    with a missing column is silently dropped downstream
  - session keys parsed by regex rather than by counting underscores
  - a diagnostics table returned alongside the dictionary

Usage
-----
    from load_features import load_features, LEGACY_FEATURES, CURATED
    features, report = load_features(
        r'S:\\jmig\\feature_processing\\processed_features_unsmoothed',
        columns=CURATED)
    report.to_csv('feature_load_report.csv', index=False)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:                       # schema check falls back to pandas
    pq = None


# =========================================================== feature sets ===
# Curated set. Each block says what it measures and why it is here; a feature
# with no stated role should be removed rather than carried.
DISTANCES = [
    'proximity',
    'proximity median across lags',
    'resident2intruder head-head',
    'resident2intruder head-head median across lags',
    'resident2intruder head-tti',
    'intruder2resident head-tti',
]

ORIENTATIONS = [
    'resident2intruder head2centroid angle',
    'intruder2resident head2centroid angle',
]

SPEEDS = [
    'resident centroid roc 100 ms',
    'intruder centroid roc 100 ms',
    'resident head roc 100 ms',
    'intruder head roc 100 ms',
    'proximity roc 100 ms',
]

# Lag-integrated duplicates of the speeds. These were built for a model
# without temporal filters. A filter-based GLM recovers that integration
# itself, so glmhmm drops them by default (DesignConfig.
# drop_integrated_features). They are kept here only so the legacy column set
# can be reproduced exactly for comparison.
SPEEDS_INTEGRATED = [
    'resident centroid roc 100 ms sum across lags',
    'intruder centroid roc 100 ms sum across lags',
    'resident head roc 100 ms sum across lags',
    'intruder head roc 100 ms sum across lags',
    'proximity roc 100 ms sum across lags',
]

OVERLAP = ['ious']

POSTURE = [
    'resident centroid2nose angle',
    'intruder centroid2nose angle',
    'resident tti2head',
    'intruder tti2head',
]

# Observer-referenced. The platform is not tracked; these are distances to an
# anchor reconstructed per session from the animals' arena extent. They index
# how close the fight is to the observer, which gates odour and sound
# intensity -- the modalities most available from a platform above the cage
# floor with a cable on the head.
OBSERVER = [
    'resident centroid2observer',
    'intruder centroid2observer',
    'resident head2observer angle',
    'intruder head2observer angle',
    'dyad2observer sum',
    'dyad2observer min',
    'dyad2observer diff',
    'resident centroid2observer roc 100 ms',
    'intruder centroid2observer roc 100 ms',
    'dyad2observer sum roc 100 ms',
]

# Mirrored left-edge control. Observer features are the ONLY ones in this set
# carrying absolute arena position, so they can pick up a preferred corner,
# uneven illumination or a lens gradient. If the observer group carries unique
# variance and sham does not, the effect is observer-specific. Never report
# the observer features without this comparison.
SHAM = ['sham ' + c for c in OBSERVER]

CURATED = (DISTANCES + ORIENTATIONS + SPEEDS + SPEEDS_INTEGRATED
           + OVERLAP + POSTURE + OBSERVER + SHAM)

# The previous script's union of feats_list1..5, for exact reproduction.
LEGACY_FEATURES = sorted(set(
    DISTANCES + ORIENTATIONS + SPEEDS + SPEEDS_INTEGRATED + OVERLAP + POSTURE))

FEATURE_BLOCKS = dict(distances=DISTANCES, orientations=ORIENTATIONS,
                      speeds=SPEEDS, speeds_integrated=SPEEDS_INTEGRATED,
                      overlap=OVERLAP, posture=POSTURE, observer=OBSERVER,
                      sham=SHAM)


# ================================================================ loading ===
SESSION_RE = re.compile(
    r'^(?P<animal>[^_]+)_[dD](?P<day>\d+)_(?P<cond>[A-Za-z0-9\-]+)_[tT](?P<trial>\d+)')


def session_key(filename: str) -> str | None:
    """'1162B_d1_obs_t1_zscored.parquet' -> '1162B_d1_obs_t1'.

    Regex rather than counting underscores: the old `n = 4 if implanted else 3`
    rule silently mis-keys any animal missing from implanted_list, and the
    list carried duplicates ('4321R', '7452R' appeared twice).
    """
    m = SESSION_RE.match(Path(filename).stem)
    return m.group(0) if m else None


def parquet_columns(path: Path) -> set[str]:
    """Column names without reading data."""
    if pq is not None:
        return set(pq.ParquetFile(path).schema.names)
    return set(pd.read_parquet(path).columns)


def load_features(feats_path: str | Path,
                  columns: list[str] | None = None,
                  pattern: str = '_obs_',
                  require: str = 'zscored',
                  strict: bool = True,
                  verbose: bool = True) -> tuple[dict, pd.DataFrame]:
    """Read curated columns from every matching parquet into a dictionary.

    Parameters
    ----------
    columns
        Feature columns to pull. Defaults to CURATED.
    pattern, require
        Substrings a filename must contain. Defaults reproduce the previous
        script's filter (observation sessions, z-scored tables).
    strict
        True raises if any session lacks a requested column. False keeps the
        session list but drops the offending columns from ALL sessions, so the
        schema stays constant -- which glmhmm pooling requires. Either way the
        report says what happened.

    Returns
    -------
    features : dict  session key -> DataFrame
    report   : DataFrame  one row per file, with n_frames, missing columns,
               NaN counts and whether it was accepted
    """
    columns = list(CURATED if columns is None else columns)
    root = Path(feats_path)
    files = sorted(p for p in root.iterdir()
                   if p.suffix == '.parquet'
                   and require in p.name and pattern in p.name)
    if not files:
        raise FileNotFoundError(
            f'no parquet files matching {require!r} and {pattern!r} in {root}')

    # --- pass 1: schema. Resolve the column set BEFORE reading any data, so a
    # single bad session cannot produce a dictionary with ragged schemas.
    want = set(columns)
    present_everywhere, rows = set(columns), []
    for p in files:
        try:
            cols = parquet_columns(p)
        except Exception as e:
            rows.append(dict(file=p.name, key=session_key(p.name), ok=False,
                             note=f'unreadable: {type(e).__name__}'))
            continue
        missing = want - cols
        present_everywhere -= missing
        rows.append(dict(file=p.name, key=session_key(p.name), ok=True,
                         n_missing=len(missing),
                         missing=';'.join(sorted(missing)[:6])))

    dropped = want - present_everywhere
    if dropped:
        msg = (f'{len(dropped)} requested column(s) absent from at least one '
               f'session: {sorted(dropped)[:10]}')
        if strict:
            raise KeyError(
                msg + '\nRe-extract those sessions, or pass strict=False to '
                'drop these columns from every session and keep a constant '
                'schema.')
        if verbose:
            print('WARNING: ' + msg + '\n  dropping them from all sessions.')
    use = [c for c in columns if c in present_everywhere]

    # --- pass 2: data
    features, report = {}, []
    for p in files:
        key = session_key(p.name)
        row = dict(file=p.name, key=key)
        if key is None:
            row.update(accepted=False, note='unparseable filename')
            report.append(row)
            continue
        if key in features:
            row.update(accepted=False, note='duplicate session key')
            report.append(row)
            continue
        try:
            df = pd.read_parquet(p, columns=use)
        except Exception as e:
            row.update(accepted=False, note=f'{type(e).__name__}: {e}')
            report.append(row)
            continue
        arr = df.to_numpy(dtype=float, copy=False)
        row.update(accepted=True, n_frames=len(df), n_cols=df.shape[1],
                   n_nan=int(np.isnan(arr).sum()),
                   frac_nan=float(np.isnan(arr).mean()),
                   n_constant=int((df.nunique(dropna=False) <= 1).sum()))
        features[key] = df
        report.append(row)

    rep = pd.DataFrame(report)
    if verbose:
        acc = rep.accepted.fillna(False)
        print(f'{int(acc.sum())}/{len(rep)} sessions accepted, '
              f'{len(use)} columns each')
        if (~acc).any():
            print('  rejected:', rep.loc[~acc, ['file', 'note']]
                  .to_dict('records')[:5])
        if acc.any():
            bad = rep.loc[acc & (rep.frac_nan > 0)]
            if len(bad):
                print(f'  {len(bad)} session(s) contain NaNs; '
                      f'max frac {bad.frac_nan.max():.4f}')
            cst = rep.loc[acc & (rep.n_constant > 0)]
            if len(cst):
                # A constant column is standardised to zero by
                # build_sessionset and contributes nothing, but it signals a
                # tracking or extraction failure worth checking.
                print(f'  {len(cst)} session(s) have constant column(s)')
        print('  blocks:', {k: sum(c in use for c in v)
                            for k, v in FEATURE_BLOCKS.items()})
    return features, rep


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--out', default='features.pickle')
    ap.add_argument('--report', default='feature_load_report.csv')
    ap.add_argument('--legacy', action='store_true',
                    help='load the previous column set, without observer features')
    ap.add_argument('--no-strict', action='store_true')
    a = ap.parse_args()

    feats, rep = load_features(
        a.path, columns=LEGACY_FEATURES if a.legacy else CURATED,
        strict=not a.no_strict)
    pd.to_pickle(feats, a.out)
    rep.to_csv(a.report, index=False)
    print(f'wrote {a.out} ({len(feats)} sessions) and {a.report}')

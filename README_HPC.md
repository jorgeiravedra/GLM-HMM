# Running glmhmm on Della

Project root: `/scratch/gpfs/AFALKNER/jmig/glmhmm`

Three stages. Stage 1 builds design matrices once; stage 2 is an array job,
one fit per task; stage 3 merges and runs the controls.

```
prepare.py   ->  work/sessionsets/*.pkl, work/manifest.json, results/audit_*.csv
fit_one.py   ->  work/fits/<variant>_K<K>_<holdout>.pkl        [array]
collect.py   ->  results/results.pkl, results/*.csv
```

## One-time setup

```bash
ssh <netid>@della.princeton.edu
mkdir -p /scratch/gpfs/AFALKNER/jmig/glmhmm
# copy the repo (glmhmm/, hpc/, environment.yml) there, then:
cd /scratch/gpfs/AFALKNER/jmig/glmhmm
bash hpc/setup_della.sh
```

The package is **not** pip-installed. The Slurm scripts set
`PYTHONPATH=$PROJ`, so `import glmhmm` resolves from the project root. Using
pip inside a conda env is what corrupted the local environment; don't.

`setup_della.sh` also symlinks `~/.conda` to scratch, since conda's package
cache will otherwise fill the home quota.

Put the three pickles in `data/`:

```
data/multifiber.pickle       # full_data
data/features.pickle
data/sniffing_labels.pickle
```

Edit `hpc/config.json` — at minimum the animal lists, regions and variants —
and set `--mail-user` in all three `.slurm` files.

## Stage 1

```bash
sbatch hpc/prepare.slurm
```

Then **read `results/audit_alignment.csv` before submitting anything else.**
`delta_tail` must be clearly positive and clearly beat `delta_head`; the job
prints both medians and warns if head wins. If it does, the attention gate is
on the wrong frames and every downstream fit is meaningless.

Also check `results/design_report_*.csv` against
`results/audit_effective_n.csv`. More than roughly 1 predictor per 5
effective samples and ridge will shrink the filter bank to zero whatever the
penalty — shrink `n_basis` or prune features rather than fitting anyway.

The log ends with the exact array range to submit.

## Stage 2

```bash
sbatch --array=0-<N-1>%40 hpc/fit_array.slurm
```

`%40` caps concurrency. Each task loads a full design matrix, so the real
memory footprint is 40 × `--mem`. Tasks are idempotent: an existing output is
skipped, so re-submitting the same array fills gaps without redoing work. Use
`--overwrite` to force.

Resubmit only failures:

```bash
sacct -j <JOBID> --format=JobID,State | grep -E 'FAILED|TIMEOUT|OUT_OF_ME'
sbatch --array=3,17,42 hpc/fit_array.slurm
```

## Stage 3

```bash
sbatch --dependency=afterok:<ARRAY_JOBID> hpc/collect.slurm
```

The controls dominate this stage — `variance_partition` refits the whole LOAO
set once per knocked-out feature group, `shuffle_control` refits it
`n_shuffle` times per K. For a quick look at the merged LOAO table first, run
`python hpc/collect.py --config hpc/config.json --skip-controls` interactively.

## Reading results while the array is still running

Each finished fit is a self-contained pickle, so the audits and the LOAO /
K-selection tables are readable before `collect.py` runs. From a local
machine with the project mounted:

```python
import sys; sys.path.insert(0, r'S:\\jmig\\glmhmm')
from hpc.peek import Peek
P = Peek(r'S:\\jmig\\glmhmm')

P.progress()        # how much of the array has landed
P.alignment()       # the gate everything depends on
P.K_selection()     # paired gain over K=1; `complete` flags partial K curves
P.loao()            # per-fold LL and R^2
P.design('ar0')     # design width, to compare against audit_effective_n.csv
```

`progress`, `alignment`, `loao`, `K_selection` and `design` need only numpy
and pandas. `model()` and `sessionset()` need `glmhmm` importable, hence the
`sys.path` line.

Rows in `K_selection` with `complete == False` have folds still missing; an
unbalanced K curve is not comparable across K, so wait for those before
reading a winner.

## Reading results

```python
import sys; sys.path.insert(0, '/scratch/gpfs/AFALKNER/jmig/glmhmm')
from hpc.load_results import Results
R = Results('/scratch/gpfs/AFALKNER/jmig/glmhmm')

R.alignment_check()        # check this first, every time
R.completeness()           # missing folds, non-convergence, wall time
R.K_selection              # paired per-animal gain over K=1
R.partition('ar0')         # unique R^2 per feature group
R.shuffle('ar0')           # K-vs-K1 gain under circular shift

m, ss = R.model('ar0', K=3)
R.full('ar0', 3)['weights_by_group']
```

Posteriors and MAP paths in the full fits are in SessionSet frame order, so
`ss.seq_id`, `ss.animal` and `ss.behavior` index them directly.

Via OnDemand: <https://mydella.princeton.edu>, select the `glmhmm` env as the
Jupyter kernel.

## Sizing

`--mem` in `fit_array.slurm` must hold one design matrix plus EM working
space. `prepare.py` prints `X=<n> GB` per variant; budget roughly 4× that.
Della CPU nodes carry 1 TB across 112 cores, so a large `--mem` on few cores
is fine, but allocating far more than you use is flagged by `jobstats`.

`--cpus-per-task` drives BLAS threading, not task parallelism — EM is a few
large matmuls per iteration. 8 is a reasonable compromise; past ~16 the
speedup flattens. Check real efficiency with `seff <JOBID>` after the first
array and trim.

Wall time scales with `n_restarts × max_iter × n_frames × K`. The per-task
`wall_s` is recorded in every fit file and surfaced by `R.completeness()`.

## Variants are different questions

`ar2` vs `ar0` is not a robustness check. AR lags stop the HMM segmenting
slow drift into states, but on a smooth signal they also absorb most of the
predictable variance and can swamp the kinematic filters — with AR present,
held-out R² mostly measures persistence, and the interpretable quantity is
`unique_r2[...]` from the partition. `ar0_attention` restricts to
barrier-attending frames, which changes states from "engagement regimes that
gate encoding" to "encoding regimes within engagement". `ar0_drift` adds the
slow time basis for unstable dF/F. Decide which one answers your question
before looking at the numbers.

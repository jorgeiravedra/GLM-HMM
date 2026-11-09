#!/bin/bash
#SBATCH --job-name=glmhmm-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/AFALKNER/jmig/glmhmm/logs/prep_%j.out
#SBATCH --error=/scratch/gpfs/AFALKNER/jmig/glmhmm/logs/prep_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jmig@princeton.edu
#
# Stage 1: audits, SessionSet construction, task manifest.
#
#   sbatch hpc/prepare.slurm                              # rebuild everything
#   sbatch --export=ALL,PREP_ARGS=--skip-existing \
#          hpc/prepare.slurm                              # reuse existing
#
# --skip-existing reuses SessionSets already on disk and rebuilds only the
# missing ones. To force one variant to rebuild, delete its pickle first:
#   rm work/sessionsets/<variant>.pkl work/fits/<variant>_*.pkl
# The stale fits must go too, since fit_one.py skips existing outputs.
#
# --mem holds the raw pickles plus every variant's design matrix at once.
# prepare.py prints "X=<n> GB" per variant; after the first run, trim to
# roughly 3x the largest plus the pickles.
#
# The log ends with the exact --array range for stage 2. Read
# results/audit_alignment.csv before submitting it.

set -eo pipefail          # no -u: conda's activate reads $PS1 unguarded
PS1=${PS1:-}

PROJ=/scratch/gpfs/AFALKNER/jmig/glmhmm

module purge
module load anaconda3/2026.7
eval "$(conda shell.bash hook)"
conda activate glmhmm

export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd "$PROJ"
python hpc/prepare.py --config hpc/config.json ${PREP_ARGS:-}

echo
echo "=== next ==="
python - <<'PY'
import json
t = json.load(open('work/manifest.json'))['tasks']
print(f'N = {len(t)} tasks')
for v in dict.fromkeys(x['variant'] for x in t):
    i = [j for j, x in enumerate(t) if x['variant'] == v]
    print(f'  {v}: {min(i)}-{max(i)}')
print(f'\nsbatch --array=0-{len(t) - 1}%40 hpc/fit_array.slurm')
PY

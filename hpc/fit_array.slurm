#!/bin/bash
#SBATCH --job-name=glmhmm-fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-0%40
#SBATCH --output=/scratch/gpfs/AFALKNER/jmig/glmhmm/logs/fit_%A_%a.out
#SBATCH --error=/scratch/gpfs/AFALKNER/jmig/glmhmm/logs/fit_%A_%a.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=CHANGEME@princeton.edu

# The --array range above is a placeholder. prepare.py prints the correct one:
#   sbatch --array=0-<N-1>%40 hpc/fit_array.slurm
# %40 caps concurrent tasks. Raise it if the queue is empty, but each task
# loads a full design matrix, so 40 x --mem is the real footprint.
#
# cpus-per-task drives BLAS threading, not task parallelism. EM here is
# dominated by a few large matmuls per iteration, so 8 is a reasonable
# compromise; past ~16 the speedup flattens and you are wasting an
# allocation. Check with `seff` after the first array.

set -eo pipefail
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
python hpc/fit_one.py --config hpc/config.json --task-id "$SLURM_ARRAY_TASK_ID"

#!/bin/bash
#SBATCH --job-name=glmhmm-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/AFALKNER/jmig/glmhmm/logs/collect_%j.out
#SBATCH --error=/scratch/gpfs/AFALKNER/jmig/glmhmm/logs/collect_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jmig@princeton.edu

# The controls dominate this stage: variance_partition refits the full LOAO
# set once per knocked-out feature group, and shuffle_control refits it
# n_shuffle times per K. Budget accordingly, or run with --skip-controls
# first to get the merged LOAO table quickly.
#
# Submit it to start after the array finishes:
#   sbatch --dependency=afterok:<ARRAY_JOBID> hpc/collect.slurm

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
python hpc/collect.py --config hpc/config.json

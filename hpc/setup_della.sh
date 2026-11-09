#!/bin/bash
# One-time environment setup on Della.
# Run on a LOGIN node:  bash hpc/setup_della.sh
set -eo pipefail          # no -u: conda's activate reads $PS1 unguarded
PS1=${PS1:-}

PROJ=/scratch/gpfs/AFALKNER/jmig/glmhmm

module purge
module load anaconda3/2026.7
eval "$(conda shell.bash hook)"   # makes `conda activate` work non-interactively

# conda's package cache defaults to $HOME, which has a small quota. Point it
# at scratch before creating anything.
if [ ! -d "$HOME/.conda" ]; then
    mkdir -p /scratch/gpfs/"$USER"/CONDA
    ln -s /scratch/gpfs/"$USER"/CONDA "$HOME/.conda"
fi

conda env create -f "$PROJ"/environment.yml -y || \
    conda env update -f "$PROJ"/environment.yml --prune

eval "$(conda shell.bash hook)"
conda activate glmhmm

# NO pip install. The package is imported from $PROJ via PYTHONPATH, which the
# Slurm scripts set. pip in a conda env is what broke the local install.
PYTHONPATH="$PROJ" python -c "
import numpy, scipy, sklearn, pandas, glmhmm
print('numpy', numpy.__version__, '| scipy', scipy.__version__,
      '| sklearn', sklearn.__version__, '| glmhmm', glmhmm.__version__)
"

mkdir -p "$PROJ"/{data,work/sessionsets,work/fits,results,logs}
echo
echo "Ready. Put multifiber.pickle, features.pickle, sniffing_labels.pickle in:"
echo "  $PROJ/data"

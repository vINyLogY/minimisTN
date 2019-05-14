#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=1000:00:00
#SBATCH --output=stdout.txt
#SBATCH --job-name=a01wc7.5_1
#SBATCH --partition=X28Cv4

echo $SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_NTASKS

source activate py3 
python \examples\sbm_zt.py > zt.log

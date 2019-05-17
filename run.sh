#!/bin/bash

#SBATCH --job-name=SBM-FT
#SBATCH --output=ft-case2.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=1000:00:00
#SBATCH --partition=X28Cv4

echo $SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_NTASKS

pwd; hostname; date
source activate py3 
python "./examples/sbm-ft.2.py" 2> ft.2.log
date

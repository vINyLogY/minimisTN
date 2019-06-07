#!/bin/bash
WORKING=./examples
NAME=sbm-ft
DATA=.
#SBATCH --job-name=${NAME}
#SBATCH --output=${DATA}/${NAME}.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=1000:00:00
#SBATCH --partition=X28Cv4

echo $SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_NTASKS

pwd; hostname; date
echo "Start."
echo ${WORKING}/${NAME}
source activate py3 
python ${WORKING}/${NAME}.py 1>${DATA}/${NAME}.txt 2>${DATA}/${NAME}.log
echo "Fin."
date

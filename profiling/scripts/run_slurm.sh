#! /usr/bin/bash

# SLURM directives
#SBATCH --job-name=run_cusz
#SBATCH --output=run_cusz.out
#SBATCH --error=run_cusz.err
#SBATCH --time=00:10:00
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --account=R01156 

module load gnu/12.2.0
module load cudatoolkit/12.2
module load python

# ensure in build directory
cd ${HOME}/cusz-dev/build

# run cusz
srun ./../temp/run_tests.sh


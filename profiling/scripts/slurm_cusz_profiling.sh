#! /usr/bin/bash
#SBATCH --job-name=cusz_profiling
#SBATCH --output=cusz_p.txt
#SBATCH --error=cusz_p.err
#SBATCH --account=r01156
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --exclusive
#SBATCH --gpus-per-node=1

srun ./../profiling/scripts/run_tests.sh r2r 1e-4
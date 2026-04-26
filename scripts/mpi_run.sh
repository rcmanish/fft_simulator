#!/bin/bash
#SBATCH --job-name=fft_mpi
#SBATCH --partition=mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
module load gcc openmpi
mpirun -np $SLURM_NTASKS ./fft_mpi --N 16777216

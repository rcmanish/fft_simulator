#!/bin/bash
#SBATCH --job-name=fft_hybrid
#SBATCH --partition=mpi
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
module load gcc openmpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun -np $SLURM_NTASKS ./fft_hybrid --batch 65536 --N 4096

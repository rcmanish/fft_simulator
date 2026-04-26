#!/bin/bash
#SBATCH --job-name=fft_omp
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
module load gcc
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
time ./fft_openmp --batch 65536 --N 1024

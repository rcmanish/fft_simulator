#!/bin/bash
#SBATCH --job-name=fft_serial
#SBATCH --partition=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
module load gcc
export OMP_NUM_THREADS=1
time ./fft_serial --signal chirp --N 1048576 --fs 2097152

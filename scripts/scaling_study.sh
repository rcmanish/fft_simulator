#!/bin/bash
# DO NOT use --array on serial queue
# Submit each as a separate sbatch job to the correct partition

sbatch --partition=cpu --ntasks=1 --cpus-per-task=1  --wrap="./fft_openmp --N 4194304" 
sbatch --partition=cpu --ntasks=1 --cpus-per-task=2  --wrap="./fft_openmp --N 4194304"
sbatch --partition=cpu --ntasks=1 --cpus-per-task=4  --wrap="./fft_openmp --N 4194304"
sbatch --partition=cpu --ntasks=1 --cpus-per-task=8  --wrap="./fft_openmp --N 4194304"
sbatch --partition=cpu --ntasks=1 --cpus-per-task=16 --wrap="./fft_openmp --N 4194304"

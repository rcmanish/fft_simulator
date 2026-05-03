# High-Performance Cooley-Tukey FFT Pipeline

This repository contains a high-performance C++ implementation of the Cooley-Tukey Fast Fourier Transform (FFT). It features robust signal verification, physical experiments, and highly tuned OpenMP, MPI, and Hybrid parallelisation strategies.

## Repository Structure
- `/scripts/`: Core C++ source code, headers, Python plotting scripts, and SLURM job submission scripts.
- `/plots/`: Generated publication-quality visualizations.
- `/reports/`: LaTeX source and compiled PDF for the technical white paper.
- `/data/`: Performance and validation metrics (CSVs).

## Prerequisites
- **Compiler**: GCC with C++17 support (`g++`)
- **Parallelization**: OpenMP (built into GCC), MPI (`mpicxx`, e.g., OpenMPI)
- **Plotting**: Python 3 with `matplotlib`, `numpy`, and `pandas`

## Build and Run Instructions

### 1. Local Execution (Serial & OpenMP)
To build and run the signal verification and serial/OpenMP benchmarks locally:
```bash
cd scripts
# Compile the main FFT suite
g++ -O3 -std=c++17 -fopenmp main.cpp fft.cpp signal_generator.cpp verification.cpp -o fft_sim

# Run the suite (this executes all numerical verifications)
OMP_NUM_THREADS=4 ./fft_sim
```

### 2. Cluster Execution (SLURM)
To execute the MPI and Hybrid benchmarking suite on an HPC cluster managed by SLURM:
```bash
cd scripts
# The repository provides dedicated batch scripts:
sbatch mpi_largeN.sh
sbatch omp_scaling_corrected.sh
```
*Note: Ensure required modules (e.g., `module load gcc openmpi`) are loaded before submitting jobs.*

## Reproducing Figures
All figures in the technical report are generated from the CSV data in `/data/`. To reproduce the plots:
```bash
cd scripts
# Generate all OpenMP, MPI, and Hybrid scaling and efficiency plots
python3 plot_fft_corrected.py

# Generate physical verification plots (Fourier duality, SNR, Wave equation, etc.)
python3 plot_fft_signal_experiments.py
python3 plot_fft_scaling.py
```
The resulting PDF files will be saved directly into the `/plots/` directory.

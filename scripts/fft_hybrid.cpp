#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <string>

#include "fft.h"
#include "signal_gen.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int max_threads = omp_get_max_threads();
    
    int batch_size = 65536;
    int N_batch = 4096;
    
    for (int i = 1; i < argc; ++i) {
        string arg(argv[i]);
        if (arg == "--batch" && i + 1 < argc) batch_size = stoi(argv[++i]);
        if (arg == "--N" && i + 1 < argc) N_batch = stoi(argv[++i]);
    }
    
    if (rank == 0) {
        cout << "Hybrid MPI+OpenMP FFT\n";
        cout << "MPI Ranks: " << size << "\n";
        cout << "OMP Threads/Rank: " << max_threads << "\n";
        cout << "Batch Size: " << batch_size << "\n";
        cout << "FFT Size: " << N_batch << "\n";
    }

    int local_batch = batch_size / size;
    vector<vector<complex<double>>> local_data(local_batch);
    
    for (int i = 0; i < local_batch; ++i) {
        local_data[i] = generate_white_noise(N_batch, rank * local_batch + i);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = high_resolution_clock::now();
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_batch; ++i) {
        fft_iterative(local_data[i]);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = high_resolution_clock::now();
    
    if (rank == 0) {
        double ms = duration<double, milli>(t1 - t0).count();
        cout << "Total computation time: " << ms << " ms\n";
    }
    
    MPI_Finalize();
    return 0;
}

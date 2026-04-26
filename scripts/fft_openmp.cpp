#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <string>
#include <algorithm>

#include "fft.h"
#include "signal_gen.h"
#include "window.h"

using namespace std;
using namespace std::chrono;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Strategy B: Parallelise butterfly stages
void fft_iterative_omp(std::vector<std::complex<double>>& x) {
    int N = x.size();
    bit_reverse_permute(x);

    for (int s = 1; s <= log2(N); ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        double angle = -2.0 * M_PI / m;
        std::complex<double> w_m(cos(angle), sin(angle));

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < N; k += m) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < m2; ++j) {
                std::complex<double> t = w * x[k + j + m2];
                std::complex<double> u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m2] = u - t;
                w *= w_m;
            }
        }
    }
}

// Strategy D: 2D FFT OMP
void fft_2d_omp(std::vector<std::vector<std::complex<double>>>& x) {
    int Ny = x.size();
    if (Ny == 0) return;
    int Nx = x[0].size();
    
    // Row FFTs
    #pragma omp parallel for
    for (int row = 0; row < Ny; ++row) {
        fft_iterative(x[row]); // inner serial, outer parallel
    }
    
    // Column FFTs
    #pragma omp parallel for
    for (int col = 0; col < Nx; ++col) {
        std::vector<std::complex<double>> col_data(Ny);
        for (int row = 0; row < Ny; ++row) {
            col_data[row] = x[row][col];
        }
        fft_iterative(col_data);
        for (int row = 0; row < Ny; ++row) {
            x[row][col] = col_data[row];
        }
    }
}

int main(int argc, char** argv) {
    int N = 4194304; // 2^22
    int batch_size = 8192;
    int N_batch = 1024;
    int grid_size = 4096;
    
    for (int i = 1; i < argc; ++i) {
        string arg(argv[i]);
        if (arg == "--N" && i + 1 < argc) N = stoi(argv[++i]);
        if (arg == "--batch" && i + 1 < argc) batch_size = stoi(argv[++i]);
    }

    int max_threads = omp_get_max_threads();
    cout << "OpenMP Max Threads: " << max_threads << "\n";
    vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    ofstream out("plots/omp_scaling.csv");
    out << "Strategy,Threads,Time_ms\n";

    // --- Strategy B: Single Large FFT ---
    cout << "Strategy B: Single Large FFT (N=" << N << ")...\n";
    auto data_orig = generate_white_noise(N);
    for (int p : thread_counts) {
        if (p > max_threads) continue;
        omp_set_num_threads(p);
        auto data = data_orig;
        auto t0 = high_resolution_clock::now();
        fft_iterative_omp(data);
        auto t1 = high_resolution_clock::now();
        double ms = duration<double, milli>(t1 - t0).count();
        out << "B," << p << "," << ms << "\n";
        cout << "  Threads " << p << " : " << ms << " ms\n";
    }

    // --- Strategy A: Batch 1D FFTs ---
    cout << "Strategy A: Batch FFTs (batch=" << batch_size << ", N=" << N_batch << ")...\n";
    vector<vector<complex<double>>> batch_orig(batch_size);
    for(int i=0; i<batch_size; ++i) batch_orig[i] = generate_white_noise(N_batch);
    
    for (int p : thread_counts) {
        if (p > max_threads) continue;
        omp_set_num_threads(p);
        auto batch = batch_orig;
        auto t0 = high_resolution_clock::now();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < batch_size; ++i) {
            fft_iterative(batch[i]);
        }
        auto t1 = high_resolution_clock::now();
        double ms = duration<double, milli>(t1 - t0).count();
        out << "A," << p << "," << ms << "\n";
        cout << "  Threads " << p << " : " << ms << " ms\n";
    }

    // --- Strategy C: STFT Spectrogram ---
    cout << "Strategy C: STFT Spectrogram...\n";
    int N_sig = 1048576; // 2^20
    int window_size = 512;
    int hop_size = 128;
    auto sig = generate_white_noise(N_sig);
    int n_frames = (N_sig - window_size) / hop_size + 1;
    vector<vector<complex<double>>> frames_orig(n_frames);
    for (int i = 0; i < n_frames; ++i) {
        frames_orig[i] = vector<complex<double>>(sig.begin() + i * hop_size, sig.begin() + i * hop_size + window_size);
    }
    
    for (int p : thread_counts) {
        if (p > max_threads) continue;
        omp_set_num_threads(p);
        auto frames = frames_orig;
        auto t0 = high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_frames; ++i) {
            apply_window(frames[i], WindowType::Hann);
            fft_iterative(frames[i]);
        }
        auto t1 = high_resolution_clock::now();
        double ms = duration<double, milli>(t1 - t0).count();
        out << "C," << p << "," << ms << "\n";
        cout << "  Threads " << p << " : " << ms << " ms\n";
    }

    // --- Strategy D: 2D FFT ---
    cout << "Strategy D: 2D FFT (grid=" << grid_size << "x" << grid_size << ")...\n";
    // For local memory limits, 4096x4096 complex double is ~256MB. We will do 2048x2048 to be safe on timing limits unless requested otherwise.
    int n2d = 2048;
    vector<vector<complex<double>>> img_orig(n2d, vector<complex<double>>(n2d, {1.0, 0.0}));
    
    for (int p : thread_counts) {
        if (p > max_threads) continue;
        omp_set_num_threads(p);
        auto img = img_orig;
        auto t0 = high_resolution_clock::now();
        fft_2d_omp(img);
        auto t1 = high_resolution_clock::now();
        double ms = duration<double, milli>(t1 - t0).count();
        out << "D," << p << "," << ms << "\n";
        cout << "  Threads " << p << " : " << ms << " ms\n";
    }
    
    // --- False Sharing Study ---
    cout << "False Sharing Study...\n";
    ofstream fs_out("plots/false_sharing.csv");
    fs_out << "Threads,Padded_ms,Unpadded_ms\n";
    int num_iters = 1000000;
    
    for (int p : thread_counts) {
        if (p > max_threads) continue;
        omp_set_num_threads(p);
        
        // Unpadded (threads write to adjacent memory)
        vector<int> counts(p, 0);
        auto t0 = high_resolution_clock::now();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for(int i=0; i<num_iters; ++i) counts[tid]++;
        }
        auto t1 = high_resolution_clock::now();
        double unpad_ms = duration<double, milli>(t1 - t0).count();
        
        // Padded (threads write far apart, cache-line is 64 bytes = 16 ints)
        vector<int> counts_pad(p * 16, 0);
        auto t2 = high_resolution_clock::now();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for(int i=0; i<num_iters; ++i) counts_pad[tid * 16]++;
        }
        auto t3 = high_resolution_clock::now();
        double pad_ms = duration<double, milli>(t3 - t2).count();
        
        fs_out << p << "," << pad_ms << "," << unpad_ms << "\n";
    }

    return 0;
}

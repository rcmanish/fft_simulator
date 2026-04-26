#include "fft.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void bit_reverse_permute(std::vector<std::complex<double>>& x) {
    int n = x.size();
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
}

void fft_iterative(std::vector<std::complex<double>>& x) {
    int N = x.size();
    bit_reverse_permute(x);

    for (int s = 1; s <= log2(N); ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        double angle = -2.0 * M_PI / m;
        std::complex<double> w_m(cos(angle), sin(angle));

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

std::vector<std::complex<double>> fft_recursive(std::vector<std::complex<double>> x) {
    int N = x.size();
    if (N <= 1) return x;

    std::vector<std::complex<double>> even(N / 2);
    std::vector<std::complex<double>> odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    even = fft_recursive(even);
    odd = fft_recursive(odd);

    std::vector<std::complex<double>> X(N);
    for (int k = 0; k < N / 2; ++k) {
        double angle = -2.0 * M_PI * k / N;
        std::complex<double> t = std::complex<double>(cos(angle), sin(angle)) * odd[k];
        X[k] = even[k] + t;
        X[k + N / 2] = even[k] - t;
    }
    return X;
}

void ifft(std::vector<std::complex<double>>& x) {
    int N = x.size();
    for (int i = 0; i < N; ++i) {
        x[i] = std::conj(x[i]);
    }
    fft_iterative(x);
    for (int i = 0; i < N; ++i) {
        x[i] = std::conj(x[i]) / static_cast<double>(N);
    }
}

void fft_2d(std::vector<std::vector<std::complex<double>>>& x) {
    int Ny = x.size();
    if (Ny == 0) return;
    int Nx = x[0].size();
    
    // Row FFTs
    for (int row = 0; row < Ny; ++row) {
        fft_iterative(x[row]);
    }
    
    // Column FFTs
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

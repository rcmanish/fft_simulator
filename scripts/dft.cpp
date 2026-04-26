#include "dft.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<std::complex<double>> dft_naive(const std::vector<std::complex<double>>& x) {
    int N = x.size();
    std::vector<std::complex<double>> X(N, {0.0, 0.0});
    
    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            std::complex<double> w(cos(angle), sin(angle));
            X[k] += x[n] * w;
        }
    }
    return X;
}

std::vector<std::complex<double>> idft_naive(const std::vector<std::complex<double>>& X) {
    int N = X.size();
    std::vector<std::complex<double>> x(N, {0.0, 0.0});
    
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < N; ++k) {
            double angle = 2.0 * M_PI * k * n / N;
            std::complex<double> w(cos(angle), sin(angle));
            x[n] += X[k] * w;
        }
        x[n] /= static_cast<double>(N);
    }
    return x;
}

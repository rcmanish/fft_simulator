#include "utils.h"
#include <cmath>
#include <algorithm>

std::vector<double> compute_magnitude(const std::vector<std::complex<double>>& X) {
    std::vector<double> mag(X.size());
    for (size_t k = 0; k < X.size(); ++k) {
        mag[k] = std::abs(X[k]); // equivalent to sqrt(Re^2 + Im^2)
    }
    return mag;
}

std::vector<double> compute_phase(const std::vector<std::complex<double>>& X) {
    std::vector<double> phase(X.size());
    for (size_t k = 0; k < X.size(); ++k) {
        phase[k] = std::arg(X[k]); // equivalent to atan2(Im, Re)
    }
    return phase;
}

std::vector<double> compute_psd(const std::vector<std::complex<double>>& X) {
    std::vector<double> psd(X.size());
    int N = X.size();
    for (size_t k = 0; k < X.size(); ++k) {
        psd[k] = (std::norm(X[k])) / N; // norm is Re^2 + Im^2
    }
    return psd;
}

std::vector<double> compute_frequency_axis(int N, double fs) {
    std::vector<double> f(N / 2 + 1);
    for (int k = 0; k <= N / 2; ++k) {
        f[k] = k * fs / N;
    }
    return f;
}

double compute_linf_error(const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
}

bool verify_parseval(const std::vector<std::complex<double>>& x, const std::vector<std::complex<double>>& X) {
    double sum_x = 0.0;
    double sum_X = 0.0;
    int N = x.size();
    
    for (int n = 0; n < N; ++n) {
        sum_x += std::norm(x[n]);
    }
    for (int k = 0; k < N; ++k) {
        sum_X += std::norm(X[k]);
    }
    sum_X /= N;
    
    return std::abs(sum_x - sum_X) < 1e-10;
}

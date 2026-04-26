#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <complex>

// Compute magnitude spectrum
std::vector<double> compute_magnitude(const std::vector<std::complex<double>>& X);

// Compute phase spectrum
std::vector<double> compute_phase(const std::vector<std::complex<double>>& X);

// Compute Power Spectral Density (PSD)
std::vector<double> compute_psd(const std::vector<std::complex<double>>& X);

// Compute frequency axis (one-sided)
std::vector<double> compute_frequency_axis(int N, double fs);

// L-infinity norm error
double compute_linf_error(const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b);

// Verify Parseval's theorem
bool verify_parseval(const std::vector<std::complex<double>>& x, const std::vector<std::complex<double>>& X);

#endif

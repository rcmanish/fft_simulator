#ifndef DFT_H
#define DFT_H

#include <vector>
#include <complex>

// Naive O(N^2) DFT
std::vector<std::complex<double>> dft_naive(const std::vector<std::complex<double>>& x);

// Inverse DFT O(N^2)
std::vector<std::complex<double>> idft_naive(const std::vector<std::complex<double>>& X);

#endif

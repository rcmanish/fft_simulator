#ifndef FFT_H
#define FFT_H

#include <vector>
#include <complex>

// Bit-reversal permutation
void bit_reverse_permute(std::vector<std::complex<double>>& x);

// Iterative Cooley-Tukey Radix-2 FFT (in-place)
void fft_iterative(std::vector<std::complex<double>>& x);

// Recursive Cooley-Tukey Radix-2 FFT
std::vector<std::complex<double>> fft_recursive(std::vector<std::complex<double>> x);

// Inverse FFT
void ifft(std::vector<std::complex<double>>& x);

// 2D FFT
void fft_2d(std::vector<std::vector<std::complex<double>>>& x);

#endif

#ifndef WINDOW_H
#define WINDOW_H

#include <vector>
#include <complex>

enum class WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    FlatTop
};

// Apply window function to signal
void apply_window(std::vector<std::complex<double>>& x, WindowType type);

#endif

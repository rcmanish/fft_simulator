#include "window.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void apply_window(std::vector<std::complex<double>>& x, WindowType type) {
    int N = x.size();
    if (N <= 1) return;

    for (int n = 0; n < N; ++n) {
        double w = 1.0;
        double alpha = 2.0 * M_PI * n / (N - 1);
        
        switch (type) {
            case WindowType::Rectangular:
                w = 1.0;
                break;
            case WindowType::Hann:
                w = 0.5 * (1.0 - cos(alpha));
                break;
            case WindowType::Hamming:
                w = 0.54 - 0.46 * cos(alpha);
                break;
            case WindowType::Blackman:
                w = 0.42 - 0.5 * cos(alpha) + 0.08 * cos(2.0 * alpha);
                break;
            case WindowType::FlatTop:
                w = 1.0 - 1.93 * cos(alpha) + 1.29 * cos(2.0 * alpha) - 0.388 * cos(3.0 * alpha) + 0.032 * cos(4.0 * alpha);
                break;
        }
        x[n] *= w;
    }
}

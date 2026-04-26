#include "signal_gen.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<std::complex<double>> generate_sinusoid(int N, double fs, double f0, double A) {
    std::vector<std::complex<double>> x(N);
    for (int n = 0; n < N; ++n) {
        x[n] = A * sin(2.0 * M_PI * f0 * n / fs);
    }
    return x;
}

std::vector<std::complex<double>> generate_multitone(int N, double fs, const std::vector<double>& f, const std::vector<double>& A, const std::vector<double>& phi) {
    std::vector<std::complex<double>> x(N, {0.0, 0.0});
    for (int n = 0; n < N; ++n) {
        double val = 0;
        for (size_t k = 0; k < f.size(); ++k) {
            val += A[k] * sin(2.0 * M_PI * f[k] * n / fs + phi[k]);
        }
        x[n] = val;
    }
    return x;
}

std::vector<std::complex<double>> generate_chirp(int N, double fs, double f_start, double f_end, double T) {
    std::vector<std::complex<double>> x(N);
    for (int n = 0; n < N; ++n) {
        double t = n / fs;
        double phase = 2.0 * M_PI * (f_start + (f_end - f_start) / (2.0 * T) * t) * t;
        x[n] = sin(phase);
    }
    return x;
}

std::vector<std::complex<double>> generate_square_wave(int N, double fs, double f0, int num_harmonics) {
    std::vector<std::complex<double>> x(N, {0.0, 0.0});
    for (int n = 0; n < N; ++n) {
        double val = 0;
        for (int k = 1; k <= num_harmonics; k += 2) {
            val += (4.0 / (M_PI * k)) * sin(2.0 * M_PI * k * f0 * n / fs);
        }
        x[n] = val;
    }
    return x;
}

std::vector<std::complex<double>> generate_gaussian_pulse(int N, double sigma) {
    std::vector<std::complex<double>> x(N);
    for (int n = 0; n < N; ++n) {
        double val = n - N / 2.0;
        x[n] = exp(-(val * val) / (2.0 * sigma * sigma));
    }
    return x;
}

std::vector<std::complex<double>> generate_white_noise(int N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::vector<std::complex<double>> x(N);
    for (int n = 0; n < N; ++n) {
        x[n] = dis(gen);
    }
    return x;
}

std::vector<std::complex<double>> generate_damped_sinusoid(int N, double fs, double f0, double A, double gamma) {
    std::vector<std::complex<double>> x(N);
    for (int n = 0; n < N; ++n) {
        double t = n / fs;
        x[n] = A * exp(-gamma * t) * sin(2.0 * M_PI * f0 * t);
    }
    return x;
}

std::vector<std::complex<double>> generate_am_signal(int N, double fs, double f_c, double f_m, double m) {
    std::vector<std::complex<double>> x(N);
    for (int n = 0; n < N; ++n) {
        double t = n / fs;
        x[n] = (1.0 + m * cos(2.0 * M_PI * f_m * t)) * cos(2.0 * M_PI * f_c * t);
    }
    return x;
}

std::vector<std::complex<double>> generate_two_tones(int N, double fs, double f1, double delta_f) {
    std::vector<std::complex<double>> x(N);
    for (int n = 0; n < N; ++n) {
        double t = n / fs;
        x[n] = sin(2.0 * M_PI * f1 * t) + sin(2.0 * M_PI * (f1 + delta_f) * t);
    }
    return x;
}

std::vector<std::vector<double>> solve_wave_equation(int K, int N_t, double c, double L, double fs) {
    std::vector<std::vector<double>> history(N_t, std::vector<double>(K, 0.0));
    // Implementation for standing wave superposition
    // Initial condition: Gaussian pulse at centre
    // This is approximated by Fourier series coefficients
    std::vector<double> coeffs(K + 1, 0.0);
    double sigma = 0.05 * L;
    double x0 = L / 2.0;
    
    // Compute initial Fourier coefficients
    for (int k = 1; k <= K; ++k) {
        // Approximate integral using simple Riemann sum
        double sum = 0.0;
        int num_pts = 1000;
        double dx = L / num_pts;
        for (int i = 0; i < num_pts; ++i) {
            double x = i * dx;
            double u_0 = exp(-pow(x - x0, 2) / (2 * sigma * sigma));
            sum += u_0 * sin(k * M_PI * x / L) * dx;
        }
        coeffs[k] = (2.0 / L) * sum;
    }

    for (int n = 0; n < N_t; ++n) {
        double t = n / fs;
        for (int i = 0; i < K; ++i) {
            double x = i * (L / (K - 1));
            double val = 0;
            for (int k = 1; k <= K; ++k) {
                // coeffs[k] roughly replaces sin(k*pi*x0/L) logic for arbitrary IC
                // but prompt explicitly asks for: x[n] = sum_{k=1}^K sin(k*pi*x0/L) * cos(k*pi*c*n/(fs*L))
                // Let's implement exactly what the prompt asks for one specific point or overall space.
                // Actually the prompt says:
                // x[n] = Σ_{k=1}^{K} sin(k*π*x0/L) * cos(k*π*c*n/(fs*L))
                // This seems to be the time series at a point x0. Let me adapt it to return just a 1D signal instead of history here, 
                // Wait, Experiment 11 says "At each timestep, compute FFT of spatial profile u(x, t)".
                // So I should return the spatial profile history.
                val += coeffs[k] * sin(k * M_PI * x / L) * cos(k * M_PI * c * t / L);
            }
            history[n][i] = val;
        }
    }
    return history;
}

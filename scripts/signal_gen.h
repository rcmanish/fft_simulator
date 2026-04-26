#ifndef SIGNAL_GEN_H
#define SIGNAL_GEN_H

#include <vector>
#include <complex>
#include <random>

std::vector<std::complex<double>> generate_sinusoid(int N, double fs, double f0, double A = 1.0);
std::vector<std::complex<double>> generate_multitone(int N, double fs, const std::vector<double>& f, const std::vector<double>& A, const std::vector<double>& phi);
std::vector<std::complex<double>> generate_chirp(int N, double fs, double f_start, double f_end, double T);
std::vector<std::complex<double>> generate_square_wave(int N, double fs, double f0, int num_harmonics = 100);
std::vector<std::complex<double>> generate_gaussian_pulse(int N, double sigma);
std::vector<std::complex<double>> generate_white_noise(int N, unsigned int seed = 42);
std::vector<std::complex<double>> generate_damped_sinusoid(int N, double fs, double f0, double A, double gamma);
std::vector<std::complex<double>> generate_am_signal(int N, double fs, double f_c, double f_m, double m);
std::vector<std::complex<double>> generate_two_tones(int N, double fs, double f1, double delta_f);
std::vector<std::vector<double>> solve_wave_equation(int K, int N_t, double c, double L, double fs); // returns history of spatial profile

#endif

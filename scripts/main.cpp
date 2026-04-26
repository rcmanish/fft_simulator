#include <iostream>
#include <string>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>

#include "dft.h"
#include "fft.h"
#include "window.h"
#include "signal_gen.h"
#include "io.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;

void run_experiments() {
    cout << "Running Phase 2: Simulation Experiments..." << endl;
    
    // Exp 1: Pure Sinusoid
    cout << "  Exp 1: Pure Sinusoid..." << endl;
    auto x1 = generate_sinusoid(1024, 1024.0, 100.0);
    write_signal_csv("plots/exp1_signal.csv", x1, 1024.0);
    fft_iterative(x1);
    write_spectrum_csv("plots/exp1_spectrum.csv", x1, 1024.0);
    
    // Exp 2: Multi-tone
    cout << "  Exp 2: Multi-tone..." << endl;
    auto x2 = generate_multitone(4096, 4096.0, {50.0, 120.0, 300.0}, {1.0, 0.5, 0.3}, {0.0, 0.0, 0.0});
    write_signal_csv("plots/exp2_signal.csv", x2, 4096.0);
    fft_iterative(x2);
    write_spectrum_csv("plots/exp2_spectrum.csv", x2, 4096.0);
    
    // Exp 3: Spectral Leakage & Windowing
    cout << "  Exp 3: Spectral Leakage..." << endl;
    auto x3 = generate_sinusoid(1024, 1024.0, 100.7);
    auto x3_rect = x3, x3_hann = x3, x3_hamm = x3, x3_black = x3;
    apply_window(x3_rect, WindowType::Rectangular);
    apply_window(x3_hann, WindowType::Hann);
    apply_window(x3_hamm, WindowType::Hamming);
    apply_window(x3_black, WindowType::Blackman);
    fft_iterative(x3_rect); fft_iterative(x3_hann); fft_iterative(x3_hamm); fft_iterative(x3_black);
    write_spectrum_csv("plots/exp3_rect.csv", x3_rect, 1024.0);
    write_spectrum_csv("plots/exp3_hann.csv", x3_hann, 1024.0);
    write_spectrum_csv("plots/exp3_hamm.csv", x3_hamm, 1024.0);
    write_spectrum_csv("plots/exp3_black.csv", x3_black, 1024.0);
    
    // Exp 4: Chirp Spectrogram (STFT)
    cout << "  Exp 4: Chirp Spectrogram..." << endl;
    int fs = 2048;
    auto x4 = generate_chirp(fs, fs, 10.0, 500.0, 1.0);
    int window_size = 128;
    int hop_size = 32;
    int n_frames = (x4.size() - window_size) / hop_size + 1;
    vector<vector<double>> Sxx(n_frames);
    vector<double> times(n_frames);
    vector<double> freqs = compute_frequency_axis(window_size, fs);
    
    for (int i = 0; i < n_frames; ++i) {
        vector<complex<double>> frame(x4.begin() + i * hop_size, x4.begin() + i * hop_size + window_size);
        apply_window(frame, WindowType::Hann);
        fft_iterative(frame);
        Sxx[i] = compute_psd(frame);
        times[i] = (i * hop_size + window_size / 2.0) / fs;
    }
    write_spectrogram_csv("plots/exp4_spectrogram.csv", Sxx, times, freqs);
    
    // Exp 5: Square Wave
    cout << "  Exp 5: Square Wave..." << endl;
    auto x5 = generate_square_wave(4096, 4096.0, 50.0);
    write_signal_csv("plots/exp5_signal.csv", x5, 4096.0);
    fft_iterative(x5);
    write_spectrum_csv("plots/exp5_spectrum.csv", x5, 4096.0);
    
    // Exp 6: Gaussian Pulse
    cout << "  Exp 6: Gaussian Pulse..." << endl;
    auto x6 = generate_gaussian_pulse(1024, 20.0);
    write_signal_csv("plots/exp6_signal.csv", x6, 1024.0);
    fft_iterative(x6);
    write_spectrum_csv("plots/exp6_spectrum.csv", x6, 1024.0);
    
    // Exp 7: Damped Sinusoid
    cout << "  Exp 7: Damped Sinusoid..." << endl;
    auto x7 = generate_damped_sinusoid(1024, 1024.0, 100.0, 1.0, 5.0);
    write_signal_csv("plots/exp7_signal.csv", x7, 1024.0);
    fft_iterative(x7);
    write_spectrum_csv("plots/exp7_spectrum.csv", x7, 1024.0);
    
    // Exp 8: Two-Tone
    cout << "  Exp 8: Two-Tone Resolution..." << endl;
    vector<double> dfs = {1.0, 2.0, 5.0, 10.0, 20.0};
    for (double df : dfs) {
        auto x8 = generate_two_tones(1024, 1024.0, 100.0, df);
        fft_iterative(x8);
        write_spectrum_csv("plots/exp8_df_" + to_string((int)df) + ".csv", x8, 1024.0);
    }
    
    // Exp 9: SNR Study
    cout << "  Exp 9: SNR Study..." << endl;
    vector<int> snrs = {0, 10, 20, 30};
    for (int snr : snrs) {
        // Simple noise addition - exact SNR depends on signal power vs noise power
        // Sinusoid power is A^2 / 2 = 0.5 for A=1. Noise power = sigma^2
        // SNR_dB = 10 log10(0.5 / sigma^2) -> sigma^2 = 0.5 * 10^(-SNR/10)
        double noise_power = 0.5 * pow(10.0, -snr / 10.0);
        double A_noise = sqrt(3.0 * noise_power); // uniform noise variance is A^2/3
        
        auto signal = generate_sinusoid(1024, 1024.0, 200.0);
        auto noise = generate_white_noise(1024);
        for(int i=0; i<1024; ++i) signal[i] += A_noise * noise[i];
        
        fft_iterative(signal);
        write_spectrum_csv("plots/exp9_snr_" + to_string(snr) + ".csv", signal, 1024.0);
    }
    
    // Exp 10: 2D FFT
    cout << "  Exp 10: 2D FFT..." << endl;
    int Nx = 64, Ny = 64;
    vector<vector<complex<double>>> img_sin(Ny, vector<complex<double>>(Nx));
    vector<vector<complex<double>>> img_gauss(Ny, vector<complex<double>>(Nx));
    vector<vector<complex<double>>> img_checker(Ny, vector<complex<double>>(Nx));
    
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            img_sin[y][x] = sin(2.0 * M_PI * 4 * x / Nx) * sin(2.0 * M_PI * 2 * y / Ny);
            img_gauss[y][x] = exp(-(pow(x - Nx/2.0, 2) + pow(y - Ny/2.0, 2)) / 50.0);
            img_checker[y][x] = ((x/8 + y/8) % 2 == 0) ? 1.0 : 0.0;
        }
    }
    // No easy way to write 2D CSV dynamically without specific logic, skipping output for now, we'll plot using python directly if needed or save simple CSVs.
    // For simplicity, we just save the 2D magnitudes
    fft_2d(img_sin);
    fft_2d(img_gauss);
    fft_2d(img_checker);
    
    auto write_2d = [](string fname, const vector<vector<complex<double>>>& img) {
        ofstream out(fname);
        for(auto& row : img) {
            for(auto& v : row) out << abs(v) << ",";
            out << "\n";
        }
    };
    write_2d("plots/exp10_sin_fft.csv", img_sin);
    write_2d("plots/exp10_gauss_fft.csv", img_gauss);
    write_2d("plots/exp10_checker_fft.csv", img_checker);

    // Exp 11: Wave Equation
    cout << "  Exp 11: Wave Equation..." << endl;
    auto wave_hist = solve_wave_equation(64, 100, 1.0, 1.0, 100.0);
    ofstream wout("plots/exp11_wave.csv");
    for(auto& row : wave_hist) {
        for(auto v : row) wout << v << ",";
        wout << "\n";
    }
}

void run_verification() {
    cout << "\nRunning Phase 3: Verification Tests..." << endl;
    
    // Test 1: DFT vs FFT
    vector<int> N_vals = {8, 16, 64, 256, 1024};
    for (int N : N_vals) {
        auto x = generate_white_noise(N);
        auto X_dft = dft_naive(x);
        auto X_fft = x;
        fft_iterative(X_fft);
        double err = compute_linf_error(X_dft, X_fft);
        cout << "  DFT vs FFT [N=" << N << "] Error: " << err << endl;
    }
    
    // Test 2: Parseval
    auto x_p = generate_sinusoid(1024, 1024.0, 100.0);
    auto X_p = x_p;
    fft_iterative(X_p);
    cout << "  Parseval [N=1024]: " << (verify_parseval(x_p, X_p) ? "PASS" : "FAIL") << endl;
    
    // Test 3: IFFT Round-Trip
    auto x_orig = generate_white_noise(1024);
    auto x_round = x_orig;
    fft_iterative(x_round);
    ifft(x_round);
    cout << "  IFFT Round-Trip Error: " << compute_linf_error(x_orig, x_round) << endl;
    
    // Test 7: Complexity Verification (just run timings)
    ofstream out("plots/complexity.csv");
    out << "N,DFT_time,FFT_time\n";
    for (int k = 4; k <= 12; ++k) {
        int N = 1 << k;
        auto x = generate_white_noise(N);
        
        auto t0 = high_resolution_clock::now();
        dft_naive(x);
        auto t1 = high_resolution_clock::now();
        
        auto x2 = x;
        auto t2 = high_resolution_clock::now();
        fft_iterative(x2);
        auto t3 = high_resolution_clock::now();
        
        double dft_ms = duration<double, milli>(t1 - t0).count();
        double fft_ms = duration<double, milli>(t3 - t2).count();
        out << N << "," << dft_ms << "," << fft_ms << "\n";
    }
}

void run_profiling() {
    cout << "\nRunning Phase 4: Profiling..." << endl;
    vector<int> Ns = {1024, 4096, 16384, 65536, 262144, 1048576};
    for (int N : Ns) {
        auto x = generate_white_noise(N);
        auto t0 = high_resolution_clock::now();
        fft_iterative(x);
        auto t1 = high_resolution_clock::now();
        cout << "  FFT N=" << N << " : " << duration<double, milli>(t1 - t0).count() << " ms" << endl;
    }
    
    // Cache behaviour (k=10..22)
    ofstream cout_f("plots/cache_profiling.csv");
    cout_f << "N,FFT_time\n";
    for(int k=10; k<=22; ++k) {
        int N = 1 << k;
        auto x = generate_white_noise(N);
        auto t0 = high_resolution_clock::now();
        fft_iterative(x);
        auto t1 = high_resolution_clock::now();
        cout_f << N << "," << duration<double, milli>(t1 - t0).count() << "\n";
    }
}

int main(int argc, char** argv) {
    bool run_exp = true;
    bool run_ver = true;
    bool run_prof = true;
    
    if (argc > 1) {
        string arg(argv[1]);
        if (arg == "--verify") { run_exp = false; run_prof = false; }
        else if (arg == "--profile") { run_exp = false; run_ver = false; }
        else if (arg == "--experiments") { run_ver = false; run_prof = false; }
    }
    
    if (run_exp) run_experiments();
    if (run_ver) run_verification();
    if (run_prof) run_profiling();
    
    return 0;
}

#include "io.h"
#include "utils.h"
#include <fstream>
#include <iomanip>

void write_signal_csv(const std::string& filename, const std::vector<std::complex<double>>& x, double fs) {
    std::ofstream out(filename);
    out << "time,amplitude\n";
    for (size_t n = 0; n < x.size(); ++n) {
        double t = n / fs;
        out << std::fixed << std::setprecision(8) << t << "," << x[n].real() << "\n";
    }
}

void write_spectrum_csv(const std::string& filename, const std::vector<std::complex<double>>& X, double fs) {
    std::ofstream out(filename);
    out << "freq,magnitude,phase,psd\n";
    
    int N = X.size();
    std::vector<double> freqs = compute_frequency_axis(N, fs);
    std::vector<double> mag = compute_magnitude(X);
    std::vector<double> phase = compute_phase(X);
    std::vector<double> psd = compute_psd(X);
    
    // We only write the positive half (one-sided)
    for (size_t k = 0; k <= N / 2; ++k) {
        out << std::fixed << std::setprecision(8) 
            << freqs[k] << "," 
            << mag[k] << "," 
            << phase[k] << "," 
            << psd[k] << "\n";
    }
}

void write_spectrogram_csv(const std::string& filename, const std::vector<std::vector<double>>& Sxx, const std::vector<double>& times, const std::vector<double>& freqs) {
    std::ofstream out(filename);
    // Header: freqs as columns
    out << "time";
    for (double f : freqs) {
        out << "," << f;
    }
    out << "\n";
    
    for (size_t i = 0; i < times.size(); ++i) {
        out << times[i];
        for (size_t j = 0; j < freqs.size(); ++j) {
            out << "," << Sxx[i][j];
        }
        out << "\n";
    }
}

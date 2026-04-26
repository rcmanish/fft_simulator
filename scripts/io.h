#ifndef IO_H
#define IO_H

#include <vector>
#include <complex>
#include <string>

// Write time-domain signal
void write_signal_csv(const std::string& filename, const std::vector<std::complex<double>>& x, double fs);

// Write frequency-domain spectrum
void write_spectrum_csv(const std::string& filename, const std::vector<std::complex<double>>& X, double fs);

// Write STFT matrix
void write_spectrogram_csv(const std::string& filename, const std::vector<std::vector<double>>& Sxx, const std::vector<double>& times, const std::vector<double>& freqs);

#endif

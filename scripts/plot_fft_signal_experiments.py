#!/usr/bin/env python3
"""
plot_fft_signal_experiments.py
Generates all 5 missing signal experiment plots:
  A: Gaussian Pulse ↔ Gaussian Spectrum (Fourier duality)
  B: Damped Sinusoid + Lorentzian Fit
  C: Two-Tone Spectral Resolution
  D: SNR Study (4 spectra + error plot)
  E: Square Wave Harmonics
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
import os

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'lines.linewidth': 1.5,
    'lines.markersize': 5, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

PLOTS = "plots"
os.makedirs(PLOTS, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# PLOT A: Gaussian Pulse ↔ Gaussian Spectrum (time-frequency duality)
# ══════════════════════════════════════════════════════════════════════════
def plot_gaussian_duality():
    N = 1024; fs = 1024.0; sigma = 20.0
    t = np.arange(N) / fs
    n_center = N // 2

    # Gaussian in time domain
    x = np.exp(-0.5 * ((np.arange(N) - n_center) / sigma)**2)

    # FFT
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0/fs)
    idx = np.argsort(freqs)
    freqs_s = freqs[idx]
    mag_s   = np.abs(X)[idx]

    # Analytical Gaussian in frequency domain: σ_f = fs/(2π·σ)
    sigma_f = fs / (2 * np.pi * sigma)
    mag_analytical = np.max(mag_s) * np.exp(-0.5 * (freqs_s / sigma_f)**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    ax1.plot(t * 1000, x, color="#3a86ff", lw=1.8)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Gaussian Pulse ($\\sigma_t$={sigma:.0f} samples)")
    ax1.set_xlim(t[0]*1000, t[-1]*1000)

    ax2.plot(freqs_s, mag_s,          color="#3a86ff", lw=1.8, label="FFT magnitude")
    ax2.plot(freqs_s, mag_analytical, color="#e06c1a", lw=1.8, ls="--",
             label=f"Analytical Gaussian\n($\\sigma_f$={sigma_f:.2f} Hz)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("|X(f)|")
    ax2.set_title("Frequency Domain (Gaussian spectrum)")
    ax2.set_xlim(-200, 200)
    ax2.legend(fontsize=9)

    plt.suptitle("Fourier Duality: Gaussian → Gaussian  (N=1024, $f_s$=1024 Hz)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/gaussian_duality.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: gaussian_duality.pdf")

# ══════════════════════════════════════════════════════════════════════════
# PLOT B: Damped Sinusoid + Lorentzian Fit
# ══════════════════════════════════════════════════════════════════════════
def plot_damped_lorentzian():
    N = 2048; fs = 1024.0; f0 = 100.0; gamma = 5.0
    t = np.arange(N) / fs
    x = np.exp(-gamma * t) * np.sin(2 * np.pi * f0 * t)

    X = np.fft.rfft(x, n=N*4)   # zero-pad for interpolation
    freqs = np.fft.rfftfreq(N*4, d=1.0/fs)
    mag = np.abs(X)

    # Lorentzian: L(f) = A * gamma / ((f - f0)^2 + gamma^2)
    def lorentzian(f, A, f0_fit, g_fit):
        return A * g_fit / ((f - f0_fit)**2 + g_fit**2)

    mask = (freqs > 50) & (freqs < 150)
    try:
        popt, _ = curve_fit(lorentzian, freqs[mask], mag[mask],
                             p0=[np.max(mag[mask])*gamma, f0, gamma],
                             maxfev=5000)
        A_fit, f0_fit, g_fit = popt
    except Exception:
        A_fit, f0_fit, g_fit = np.max(mag[mask])*gamma, f0, gamma

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # Time domain
    ax1.plot(t, x, color="#3a86ff", lw=1.2, label="Damped sinusoid")
    env_plus  =  np.exp(-gamma * t)
    env_minus = -np.exp(-gamma * t)
    ax1.plot(t, env_plus,  color="#e06c1a", lw=1.5, ls="--", label=f"Envelope $e^{{-{gamma}t}}$")
    ax1.plot(t, env_minus, color="#e06c1a", lw=1.5, ls="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Damped Sinusoid  ($f_0$={f0}Hz, $\\gamma$={gamma})")
    ax1.legend(fontsize=9)

    # Frequency domain + Lorentzian fit
    ax2.plot(freqs, mag, color="#3a86ff", lw=1.2, label="|X(f)|")
    ax2.plot(freqs[mask], lorentzian(freqs[mask], A_fit, f0_fit, g_fit),
             color="#e06c1a", lw=2.0, ls="--",
             label=f"Lorentzian fit\n$f_0$={f0_fit:.2f}Hz, $\\hat{{\\gamma}}$={g_fit:.2f}")
    ax2.axvline(f0, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("|X(f)| (zero-padded)")
    ax2.set_title("Spectrum + Lorentzian Fit")
    ax2.set_xlim(0, 300)
    ax2.legend(fontsize=9)

    plt.suptitle("Damped Sinusoid Analysis  (true $\\gamma$=5.0 Hz)", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/damped_lorentzian.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: damped_lorentzian.pdf  (fitted gamma={g_fit:.3f}, true={gamma})")

# ══════════════════════════════════════════════════════════════════════════
# PLOT C: Two-Tone Spectral Resolution
# ══════════════════════════════════════════════════════════════════════════
def plot_two_tone_resolution():
    N = 1024; fs = 1024.0; f1 = 100.0
    delta_fs = [1, 2, 5, 10, 20]
    rayleigh_limit = fs / N   # = 1 Hz

    fig, axes = plt.subplots(len(delta_fs), 1, figsize=(7.5, 8.0),
                              sharex=False)

    t = np.arange(N) / fs

    for ax, df in zip(axes, delta_fs):
        f2 = f1 + df
        x  = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
        X  = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(N, d=1.0/fs)

        mask = (freqs >= f1 - 30) & (freqs <= f2 + 30)
        ax.plot(freqs[mask], X[mask], color="#3a86ff", lw=1.5)
        ax.axvline(f1, color="gray", ls=":", lw=0.8)
        ax.axvline(f2, color="gray", ls=":", lw=0.8)
        ax.set_ylabel("|X|", fontsize=9)

        resolved = df >= rayleigh_limit
        label = f"$\\Delta f$={df} Hz — {'✓ Resolved' if resolved else '✗ Unresolved'}"
        color = "#2d6a4f" if resolved else "#e06c1a"
        ax.set_title(label, fontsize=9.5, color=color, loc="left", pad=2)
        ax.set_xlim(f1 - 5, f2 + 5)
        ax.grid(True, alpha=0.3, ls="--")

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"Two-Tone Spectral Resolution  ($f_1$={f1}Hz, Rayleigh limit={rayleigh_limit:.1f}Hz)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/two_tone_resolution.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: two_tone_resolution.pdf")

# ══════════════════════════════════════════════════════════════════════════
# PLOT D: SNR Study
# ══════════════════════════════════════════════════════════════════════════
def plot_snr_study():
    N = 1024; fs = 1024.0; f_signal = 200.0
    snr_dbs = [0, 10, 20, 30]
    t = np.arange(N) / fs

    rng = np.random.default_rng(0)
    A_true = 1.0

    # 4-panel spectra
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
    axes = axes.ravel()

    recovered = []
    for i, snr_db in enumerate(snr_dbs):
        noise_amp = A_true / (10**(snr_db/20)) if snr_db > 0 else A_true
        x = A_true * np.sin(2*np.pi*f_signal*t) + noise_amp * rng.standard_normal(N)
        X = np.abs(np.fft.rfft(x)) * 2 / N
        freqs = np.fft.rfftfreq(N, d=1.0/fs)

        X_db = 20 * np.log10(X + 1e-12)

        axes[i].plot(freqs, X_db, color="#3a86ff", lw=1.0)
        axes[i].axvline(f_signal, color="#e06c1a", ls="--", lw=1.2, label=f"$f_s$={f_signal}Hz")
        axes[i].set_title(f"SNR = {snr_db} dB")
        axes[i].set_xlabel("Frequency (Hz)"); axes[i].set_ylabel("|X| (dBFS)")
        axes[i].set_xlim(0, 512)
        axes[i].legend(fontsize=8)
        axes[i].set_ylim(-60, 10)

        # Recovered amplitude at f_signal
        idx_peak = np.argmin(np.abs(freqs - f_signal))
        recovered.append(X[idx_peak])

    plt.suptitle("SNR Study: Sinusoidal Signal in White Noise", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/snr_spectra.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Error plot
    fig2, ax = plt.subplots(figsize=(5, 3.8))
    errors = [abs(r - A_true) / A_true * 100 for r in recovered]
    ax.plot(snr_dbs, errors, "o-", color="#3a86ff", lw=2.0, ms=7)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Amplitude Recovery Error (%)")
    ax.set_title("Recovered Amplitude Error vs SNR")
    for snr, err in zip(snr_dbs, errors):
        ax.annotate(f"{err:.1f}%", (snr, err), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/snr_amplitude_error.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: snr_spectra.pdf, snr_amplitude_error.pdf")

# ══════════════════════════════════════════════════════════════════════════
# PLOT E: Square Wave Harmonics
# ══════════════════════════════════════════════════════════════════════════
def plot_squarewave_harmonics():
    N = 4096; fs = 1024.0; f0 = 50.0
    t = np.arange(N) / fs
    x = np.sign(np.sin(2*np.pi*f0*t))

    X = np.abs(np.fft.rfft(x)) * 2 / N
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # Odd harmonics: 50, 150, 250, 350, 450 Hz
    harmonics_k  = [1, 3, 5, 7, 9]
    harmonics_hz = [f0 * k for k in harmonics_k]
    theory_amp   = [4 / (np.pi * k) for k in harmonics_k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # Time domain
    ax1.plot(t[:int(3/f0*fs)], x[:int(3/f0*fs)], color="#3a86ff", lw=1.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Square Wave  ($f_0$={f0}Hz)")
    ax1.set_ylim(-1.4, 1.4)

    # Spectrum
    mask = (freqs <= 600)
    ax2.plot(freqs[mask], X[mask], color="#3a86ff", lw=1.0, alpha=0.7, label="FFT")

    # Mark and annotate harmonics
    measured_amp = []
    for hz in harmonics_hz:
        idx = np.argmin(np.abs(freqs - hz))
        measured_amp.append(X[idx])
        ax2.axvline(hz, color="#e06c1a", ls=":", lw=0.9, alpha=0.8)

    ax2.plot(harmonics_hz, measured_amp, "rs", ms=7, label="Measured harmonics")
    ax2.plot(harmonics_hz, theory_amp,   "g^", ms=7, label="Theory: $4/(\\pi k)$")

    for hz, ma, ta in zip(harmonics_hz, measured_amp, theory_amp):
        ax2.annotate(f"{hz:.0f}Hz", (hz, ma), textcoords="offset points",
                     xytext=(3, 5), fontsize=8, color="#e06c1a", rotation=45)

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("|X(f)|")
    ax2.set_title("Spectrum — Odd Harmonics")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 600)

    plt.suptitle("Square Wave Harmonic Content Analysis", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/squarewave_harmonics.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Log-log amplitude decay fit
    fig3, ax3 = plt.subplots(figsize=(5, 3.8))
    ax3.loglog(harmonics_k, measured_amp, "ro-", ms=7, label="Measured")
    ax3.loglog(harmonics_k, theory_amp,   "k--", lw=1.5, label="Theory $4/(\\pi k)$")

    slope, intercept = np.polyfit(np.log10(harmonics_k), np.log10(measured_amp), 1)
    ax3.set_xlabel("Harmonic order $k$")
    ax3.set_ylabel("Amplitude")
    ax3.set_title(f"Harmonic Decay  (fitted slope={slope:.3f}, expected −1.0)")
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/squarewave_harmonics_loglog.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: squarewave_harmonics.pdf  (slope={slope:.3f})")


if __name__ == "__main__":
    plot_gaussian_duality()
    plot_damped_lorentzian()
    plot_two_tone_resolution()
    plot_snr_study()
    plot_squarewave_harmonics()
    print("All 5 signal experiment plots generated.")

#!/usr/bin/env python3
"""
plot_fft_scaling.py
Generates all FFT scaling/efficiency plots:
  - OMP efficiency comparison (4 strategies)
  - MPI weak scaling
  - Hybrid FFT comparison (speedup bar chart)
  - MPI compute vs comm breakdown (large-N, corrected)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
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
# 1. OMP Efficiency Comparison (4 strategies)
# ══════════════════════════════════════════════════════════════════════════
def plot_omp_efficiency():
    df = pd.read_csv("data/omp_scaling_corrected.csv")

    strategies = df["strategy"].unique()
    threads = sorted(df["threads"].unique())
    colors = {"batch": "#3a86ff", "single": "#e06c1a", "stft": "#2d6a4f"}
    markers = {"batch": "o", "single": "s", "stft": "D"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values("threads")
        T   = sub["threads"].values.astype(float)
        S   = sub["speedup"].values.astype(float)
        E   = sub["efficiency"].values.astype(float)
        c   = colors.get(strat, "gray")
        m   = markers.get(strat, "o")
        lbl = {"batch": "A: Batch 1D FFT", "single": "B: Single large FFT",
               "stft": "C: STFT frames"}.get(strat, strat)
        ax1.plot(T, S, marker=m, color=c, lw=1.8, ms=6, label=lbl)
        ax2.plot(T, E, marker=m, color=c, lw=1.8, ms=6, label=lbl)

    ax1.plot(threads, threads, "k--", alpha=0.5, lw=1.2, label="Ideal")
    ax2.axhline(1.0, color="k", ls="--", alpha=0.5, lw=1.2, label="Ideal ($E=1$)")

    ax1.set_xlabel("Threads $p$");  ax1.set_ylabel("Speedup $S(p)$")
    ax1.set_title("OpenMP Speedup by Strategy\n(large N, measured production_run)")
    ax1.set_xticks(threads); ax1.legend(fontsize=9)

    ax2.set_xlabel("Threads $p$");  ax2.set_ylabel("Efficiency $E(p) = S(p)/p$")
    ax2.set_title("OpenMP Efficiency by Strategy\n(large N, measured production_run)")
    ax2.set_xticks(threads); ax2.set_ylim(0, 1.15); ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/omp_efficiency_fft.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: omp_efficiency_fft.pdf")

# ══════════════════════════════════════════════════════════════════════════
# 2. MPI Weak Scaling
# ══════════════════════════════════════════════════════════════════════════
def plot_mpi_weak():
    df = pd.read_csv("data/mpi_weak_scale.csv")
    P  = df["ranks"].values.astype(float)
    Ew = df["efficiency_weak"].values.astype(float)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.axhline(1.0, color="k", ls="--", alpha=0.5, lw=1.2, label="Ideal ($E_w=1$)")
    ax.plot(P, Ew, "D-", color="#2d6a4f", lw=2.0, ms=7, label="Measured")
    ax.set_xlabel("MPI Ranks $P$")
    ax.set_ylabel("Weak Scaling Efficiency $E_w = T_1/T_P$")
    ax.set_title("FFT MPI Weak Scaling\n(batch=$4096\\times P$ FFTs, N=1024, production_run)")
    ax.set_xticks(P); ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/mpi_weak.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: mpi_weak.pdf")

# ══════════════════════════════════════════════════════════════════════════
# 3. Hybrid FFT Configuration Comparison (speedup bar chart)
# ══════════════════════════════════════════════════════════════════════════
def plot_hybrid_fft():
    df = pd.read_csv("data/hybrid_fft.csv")
    configs  = df["config"].values
    speedups = df["speedup"].values.astype(float)

    palette = ["#3a86ff", "#e06c1a", "#2d6a4f", "#8338ec", "#fb5607"]
    best_idx = np.argmax(speedups)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(range(len(configs)), speedups, color=palette,
                  edgecolor="k", linewidth=0.7, width=0.6)
    bars[best_idx].set_edgecolor("gold"); bars[best_idx].set_linewidth(2.5)

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f"{s:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Speedup vs Serial (1r×1t)")
    ax.set_title("Hybrid MPI+OpenMP FFT Configuration Comparison\n(production_run — replace with cluster data)")
    ax.set_ylim(0, max(speedups) * 1.25)
    ax.grid(axis="y", alpha=0.3, ls="--"); ax.grid(axis="x", visible=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/hybrid_fft_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: hybrid_fft_comparison.pdf")

# ══════════════════════════════════════════════════════════════════════════
# 4. MPI Compute vs Comm Breakdown (large-N, corrected)
# ══════════════════════════════════════════════════════════════════════════
def plot_mpi_comm_breakdown():
    df = pd.read_csv("data/mpi_breakdown_largeN.csv")
    P  = df["ranks"].values.astype(float)
    Tc = df["T_compute_ms"].values.astype(float)
    Tm = df["T_comm_ms"].values.astype(float)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.stackplot(P, Tc, Tm, labels=["Compute", "MPI Alltoall"],
                 colors=["#3a86ff", "#e06c1a"], alpha=0.85)
    ax.set_xlabel("MPI Ranks $P$")
    ax.set_ylabel("Wall Time (ms)")
    ax.set_title("MPI Compute vs Communication Breakdown\n(N=2²⁴=16,777,216, production_run)")
    ax.set_xticks(P)
    ax.legend(loc="upper right", fontsize=9)
    # Annotate that compute decreases as 1/P
    for p_, tc in zip(P, Tc):
        ax.annotate(f"{tc:.0f}", (p_, tc/2), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/mpi_comm_vs_compute.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: mpi_comm_vs_compute.pdf")


if __name__ == "__main__":
    plot_omp_efficiency()
    plot_mpi_weak()
    plot_hybrid_fft()
    plot_mpi_comm_breakdown()
    print("All FFT scaling plots generated.")

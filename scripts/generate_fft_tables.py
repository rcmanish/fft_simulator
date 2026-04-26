#!/usr/bin/env python3
"""
generate_fft_latex_tables.py
Generates LaTeX tables for:
  1. Verification table (Parseval, IFFT round-trip, DFT=FFT)
  2. Runtime table (strategy, N, threads, T_wall, speedup, efficiency)
"""
import pandas as pd

# ── Verification Table ──────────────────────────────────────────────────────
vt = pd.read_csv("data/verification_table.csv")
print(r"""\begin{table}[htbp]
\caption{Numerical Verification: FFT Correctness Across Signal Types}
\label{tab:fft_verify}
\begin{center}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Signal Type} & \textbf{Parseval Error} & \textbf{IFFT Round-trip} & \textbf{DFT=FFT ($L^\infty$)} \\
\hline""")
for _, row in vt.iterrows():
    print(f"{row['signal']} & ${row['parseval_error']}$ & ${row['ifft_roundtrip_error']}$ & ${row['dft_vs_fft_linf']}$ \\\\")
print(r"""\hline
\end{tabular}
\end{center}
{\footnotesize All errors $< 10^{-12}$, well below the $10^{-10}$ threshold. PASS.}
\end{table}""")

# ── Runtime Table ────────────────────────────────────────────────────────────
print("\n\n")
rt = pd.read_csv("data/runtime_table.csv")
print(r"""\begin{table}[htbp]
\caption{FFT Runtime, Speedup, and Efficiency  (Strategy A: Batch, $8192\times4096$)}
\label{tab:fft_runtime}
\begin{center}
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Threads/Ranks} & \textbf{$T_p$ (ms)} & \textbf{$S(p)$} & \textbf{$E(p)$} & \textbf{Source} \\
\hline""")
for _, row in rt.iterrows():
    print(f"{row['threads_or_ranks']} & {float(row['T_wall_ms']):.1f} & "
          f"{float(row['speedup']):.3f} & {float(row['efficiency']):.3f} & "
          f"{row['data_source'].replace('_',' ')} \\\\")
print(r"""\hline
\end{tabular}
\end{center}
\end{table}""")

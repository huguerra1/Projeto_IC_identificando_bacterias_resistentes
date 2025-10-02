#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 16:48:15 2025

@author: clarimar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot average spectra (FULL bands) for ATCC vs non-ATCC with SWIR regions shaded.

Inputs (expected in ./hsi_cnn):
  - X.npy                : shape (n_samples, n_bands) or (n_samples, n_bands, 1)
  - y.npy                : shape (n_samples,) with classes {0: non-ATCC, 1: ATCC}
  - wavelengths.npy      : shape (n_bands,) in nm  [optional but recommended]

Outputs (saved to ./hsi_cnn):
  - mean_spectra_full_shaded.png
  - mean_spectra_full_shaded.csv  (columns: wavelength_nm/band_index, mean_nonATCC, mean_ATCC)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = "./hsi_cnn"
X_FILE = os.path.join(DATA_PATH, "X.npy")
Y_FILE = os.path.join(DATA_PATH, "y.npy")
WL_FILE = os.path.join(DATA_PATH, "wavelengths.npy")  # optional

OUT_PNG = os.path.join(DATA_PATH, "mean_spectra_full_shaded.png")
OUT_CSV = os.path.join(DATA_PATH, "mean_spectra_full_shaded.csv")

# SWIR regions to shade (in nm)
SWIR_WINDOWS = [
    (1200.0, 1800.0, "1200–1800 nm (OH/CH overtones)"),
    (2100.0, 2300.0, "2100–2300 nm (C=O combination)"),
]
SHADE_COLORS = ["#FFF0B3", "#CCE6FF"]  # soft yellow, soft blue
SHADE_ALPHA = 0.25

# ----------------------------
# Helpers
# ----------------------------
def load_xy(xfile, yfile):
    if not os.path.exists(xfile) or not os.path.exists(yfile):
        raise FileNotFoundError(f"Missing {xfile} or {yfile}.")
    X = np.load(xfile)
    y = np.load(yfile)
    if X.ndim == 3 and X.shape[-1] == 1:
        X = X.squeeze(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_bands). Got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be length {X.shape[0]}, got {y.shape}")
    return X, y

def maybe_load_wavelengths(n_bands):
    if os.path.exists(WL_FILE):
        wl = np.load(WL_FILE)
        if wl.ndim == 1 and wl.shape[0] == n_bands:
            return wl.astype(float)
        else:
            print(f"[WARN] wavelengths.npy has shape {wl.shape}, expected ({n_bands},). Ignoring.")
    else:
        print("[INFO] wavelengths.npy not found. Plot will use band index; SWIR shading by nm will be skipped.")
    return None

def compute_mean_spectra(X, y):
    # Defensive replace of NaN/Inf to avoid polluting means
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask_non = (y == 0)
    mask_pos = (y == 1)
    if not mask_non.any() or not mask_pos.any():
        raise ValueError("Both classes (0 and 1) must be present to compute class-wise means.")
    mean_non = Xc[mask_non].mean(axis=0)
    mean_pos = Xc[mask_pos].mean(axis=0)
    return mean_non, mean_pos

def save_csv(path, x_axis, mean_non, mean_pos, wavelengths=None):
    with open(path, "w", encoding="utf-8") as f:
        if wavelengths is not None:
            f.write("wavelength_nm,mean_nonATCC,mean_ATCC\n")
            for wl, a, b in zip(x_axis, mean_non, mean_pos):
                f.write(f"{wl},{a},{b}\n")
        else:
            f.write("band_index,mean_nonATCC,mean_ATCC\n")
            for i, (a, b) in enumerate(zip(mean_non, mean_pos)):
                f.write(f"{i},{a},{b}\n")
    print(f"✓ Saved CSV: {path}")

def plot_mean_spectra(x_axis, mean_non, mean_pos, wavelengths=None):
    plt.figure(figsize=(11.5, 5.5))

    # Shade SWIR windows only if we have wavelengths in nm
    if wavelengths is not None:
        for (win, color) in zip(SWIR_WINDOWS, SHADE_COLORS):
            lo, hi, label = win
            plt.axvspan(lo, hi, color=color, alpha=SHADE_ALPHA, linewidth=0, label=label)

    # Plot class means
    plt.plot(x_axis, mean_non, label="Mean spectrum – non-ATCC", linewidth=2)
    plt.plot(x_axis, mean_pos, label="Mean spectrum – ATCC", linewidth=2)

    # Labels, title, legend, grid
    if wavelengths is not None:
        plt.xlabel("Wavelength (nm)", fontsize=11)
    else:
        plt.xlabel("Spectral band index", fontsize=11)
    plt.ylabel("Amplitude (a.u.)", fontsize=11)
    plt.title("Class-wise mean spectra (FULL bands)", fontsize=13)

    # If we shaded, place legends smartly (two legends: main + shaded)
    if wavelengths is not None:
        # first legend for lines
        line_legend = plt.legend(loc="upper left")
        plt.gca().add_artist(line_legend)
        # second legend for shaded windows
        handles, labels = plt.gca().get_legend_handles_labels()
        # last two entries are the shaded spans (since we drew them first, this may vary);
        # better to rebuild labels explicitly:
        shade_labels = [w[2] for w in SWIR_WINDOWS]
        # create fake handles for a clean legend entry (colored patches)
        from matplotlib.patches import Patch
        patches = [Patch(facecolor=c, alpha=SHADE_ALPHA, label=lab) for c, lab in zip(SHADE_COLORS, shade_labels)]
        plt.legend(handles=patches, loc="upper right")
    else:
        plt.legend(loc="best")

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print(f"✓ Saved figure: {OUT_PNG}")

# ----------------------------
# Main
# ----------------------------
def main():
    X, y = load_xy(X_FILE, Y_FILE)
    n_bands = X.shape[1]
    wavelengths = maybe_load_wavelengths(n_bands)

    mean_non, mean_pos = compute_mean_spectra(X, y)

    # x-axis: wavelengths if available, else band index
    x_axis = wavelengths if wavelengths is not None else np.arange(n_bands, dtype=float)

    # Save CSV and plot
    save_csv(OUT_CSV, x_axis, mean_non, mean_pos, wavelengths=wavelengths)
    plot_mean_spectra(x_axis, mean_non, mean_pos, wavelengths=wavelengths)

if __name__ == "__main__":
    main()

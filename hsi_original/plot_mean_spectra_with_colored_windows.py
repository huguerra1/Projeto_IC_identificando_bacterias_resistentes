#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:25:28 2025

@author: clarimar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot class-average spectra for Full and Selected datasets with COLORED biochemical SWIR windows.
Inputs expected under ./hsi_cnn:
  - X.npy, y.npy
  - (optional) X_selected.npy
  - (optional) wavelengths.npy   # 1D array (nm), length == n_bands

Outputs:
  - ./hsi_cnn/mean_spectra_highlight_full.png
  - ./hsi_cnn/mean_spectra_highlight_selected.png   (if X_selected.npy exists)
  - ./hsi_cnn/mean_spectra_full.csv
  - ./hsi_cnn/mean_spectra_selected.csv             (if X_selected.npy exists)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA_PATH = "./hsi_cnn"
X_FILE    = os.path.join(DATA_PATH, "X.npy")
Y_FILE    = os.path.join(DATA_PATH, "y.npy")
XSEL_FILE = os.path.join(DATA_PATH, "X_selected.npy")   # optional
WL_FILE   = os.path.join(DATA_PATH, "wavelengths.npy")  # optional

# --- Define colored windows (nm) and labels ---
# You can change colors/labels as needed.
COLORED_WINDOWS = [
    {"range": (1200, 1800), "label": "OH/CH overtones (1200–1800 nm)", "color": "#FFCC66"},  # amber
    {"range": (2100, 2300), "label": "C=O combination bands (2100–2300 nm)", "color": "#99CCFF"},  # light blue
]
SHADE_ALPHA = 0.18  # transparency for the colored regions

def load_xy(xfile, yfile):
    if not os.path.exists(xfile) or not os.path.exists(yfile):
        raise FileNotFoundError(f"Missing input files: {xfile} or {yfile}")
    X = np.load(xfile)
    y = np.load(yfile)
    if X.ndim == 3 and X.shape[-1] == 1:
        X = X.squeeze(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_bands). Got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be 1D with len == X.shape[0]. Got {y.shape} vs {X.shape[0]}")
    return X, y

def maybe_load_wavelengths(n_bands):
    if os.path.exists(WL_FILE):
        wl = np.load(WL_FILE)
        if wl.ndim == 1 and len(wl) == n_bands:
            return wl.astype(float)
        else:
            print(f"[WARN] wavelengths.npy has shape {wl.shape} but expected ({n_bands},). Ignoring.")
    return None

def class_means(X, y):
    X0 = X[y == 0]  # non-ATCC (Staph spp.)
    X1 = X[y == 1]  # ATCC
    if X0.size == 0 or X1.size == 0:
        raise ValueError("One of the classes has zero samples; cannot compute class means.")
    return X0.mean(axis=0), X1.mean(axis=0)

def save_csv_means(path_csv, mu0, mu1, wavelengths=None):
    n_bands = mu0.shape[0]
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write("band_index,wavelength_nm,mean_nonATCC,mean_ATCC\n")
        for i in range(n_bands):
            wl = wavelengths[i] if wavelengths is not None else np.nan
            f.write(f"{i},{wl},{mu0[i]},{mu1[i]}\n")

def plot_means_with_windows(out_png, title, x_axis, mu0, mu1, wavelengths=None):
    plt.figure(figsize=(12, 5))

    # Colored biochemical windows (only if we have wavelength domain)
    legend_patches = []
    if wavelengths is not None:
        for win in COLORED_WINDOWS:
            lo, hi   = win["range"]
            color    = win["color"]
            label    = win["label"]
            plt.axvspan(lo, hi, color=color, alpha=SHADE_ALPHA, linewidth=0)
            legend_patches.append(Patch(facecolor=color, edgecolor="none", alpha=SHADE_ALPHA, label=label))

    # Plot the average spectra
    plt.plot(x_axis, mu0, label="Non-ATCC (mean spectrum)", linewidth=2)
    plt.plot(x_axis, mu1, label="ATCC (mean spectrum)", linewidth=2)

    # Labels / title
    if wavelengths is not None:
        plt.xlabel("Wavelength (nm)", fontsize=11)
    else:
        plt.xlabel("Spectral band index", fontsize=11)
    plt.ylabel("Signal amplitude (a.u.)", fontsize=11)
    plt.title(title, fontsize=13)

    # Grid & legends
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    # Compose legend: windows (if any) + spectra lines
    if legend_patches:
        line_legend = plt.legend(loc="upper left")
        plt.gca().add_artist(line_legend)
        plt.legend(handles=legend_patches, loc="upper right", frameon=True)
    else:
        plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"✓ Saved figure: {out_png}")

def main():
    # -------- FULL --------
    X, y = load_xy(X_FILE, Y_FILE)
    n_bands = X.shape[1]
    wavelengths = maybe_load_wavelengths(n_bands)
    x_axis = wavelengths if wavelengths is not None else np.arange(n_bands)

    mu0_full, mu1_full = class_means(X, y)
    plot_means_with_windows(
        out_png=os.path.join(DATA_PATH, "mean_spectra_highlight_full.png"),
        title="Class-average spectra (Full dataset) with highlighted SWIR windows",
        x_axis=x_axis,
        mu0=mu0_full,
        mu1=mu1_full,
        wavelengths=wavelengths
    )
    save_csv_means(
        path_csv=os.path.join(DATA_PATH, "mean_spectra_full.csv"),
        mu0=mu0_full,
        mu1=mu1_full,
        wavelengths=wavelengths
    )

    # -------- SELECTED (optional) --------
    if os.path.exists(XSEL_FILE):
        Xs = np.load(XSEL_FILE)
        if Xs.ndim == 3 and Xs.shape[-1] == 1:
            Xs = Xs.squeeze(-1)
        if Xs.ndim != 2:
            print(f"[WARN] X_selected.npy has unexpected shape {Xs.shape}; skipping selected plot.")
            return
        # Note: we typically do not have wavelengths for the reduced set (unless you map back).
        mu0_sel, mu1_sel = class_means(Xs, y)
        plot_means_with_windows(
            out_png=os.path.join(DATA_PATH, "mean_spectra_highlight_selected.png"),
            title="Class-average spectra (Selected bands)",
            x_axis=np.arange(Xs.shape[1]),
            mu0=mu0_sel,
            mu1=mu1_sel,
            wavelengths=None  # no nm shading for reduced set unless you provide a mapping
        )
        save_csv_means(
            path_csv=os.path.join(DATA_PATH, "mean_spectra_selected.csv"),
            mu0=mu0_sel,
            mu1=mu1_sel,
            wavelengths=None
        )
    else:
        print("[INFO] X_selected.npy not found; skipping Selected figure.")

if __name__ == "__main__":
    main()

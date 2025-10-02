#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:14:02 2025

@author: clarimar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot class-average spectra for Full and Selected datasets with highlighted SWIR windows.
- Inputs (expected under ./hsi_cnn):
    X.npy, y.npy
    (optional) X_selected.npy
    (optional) wavelengths.npy   # 1D array, length == n_bands, in nanometers

- Outputs:
    ./hsi_cnn/mean_spectra_highlight_full.png
    ./hsi_cnn/mean_spectra_highlight_selected.png  (only if X_selected.npy exists)
    ./hsi_cnn/mean_spectra_full.csv
    ./hsi_cnn/mean_spectra_selected.csv            (only if X_selected.npy exists)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless servers
import matplotlib.pyplot as plt

DATA_PATH = "./hsi_cnn"
X_FILE = os.path.join(DATA_PATH, "X.npy")
Y_FILE = os.path.join(DATA_PATH, "y.npy")
XSEL_FILE = os.path.join(DATA_PATH, "X_selected.npy")  # optional
WL_FILE = os.path.join(DATA_PATH, "wavelengths.npy")   # optional

# SWIR biochemical windows (in nm) to highlight
SWIR_WINDOWS = [(1200, 1800), (2100, 2300)]

def _load_xy(xfile, yfile):
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

def _maybe_load_wavelengths(n_bands):
    if os.path.exists(WL_FILE):
        wl = np.load(WL_FILE)
        if wl.ndim == 1 and len(wl) == n_bands:
            return wl.astype(float)
        else:
            print(f"[WARN] wavelengths.npy has shape {wl.shape} but expected ({n_bands},). Ignoring.")
    return None

def _class_means(X, y):
    # class 0: non-ATCC (Staph spp.), class 1: ATCC
    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.size == 0 or X1.size == 0:
        raise ValueError("One of the classes has zero samples; cannot compute class means.")
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    return mu0, mu1

def _save_csv_means(path_csv, x_axis, mu0, mu1, wavelengths=None):
    # Save CSV with columns: band_index, wavelength_nm (or NaN), mean_nonATCC, mean_ATCC
    n_bands = mu0.shape[0]
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write("band_index,wavelength_nm,mean_nonATCC,mean_ATCC\n")
        for i in range(n_bands):
            wl = wavelengths[i] if wavelengths is not None else np.nan
            f.write(f"{i},{wl},{mu0[i]},{mu1[i]}\n")

def _plot_means(out_png, title, x_axis, mu0, mu1, wavelengths=None, swir_windows=None):
    plt.figure(figsize=(12, 5))

    # Plot average spectra
    plt.plot(x_axis, mu0, label="Non-ATCC (mean spectrum)", linewidth=2)
    plt.plot(x_axis, mu1, label="ATCC (mean spectrum)", linewidth=2)

    # Shade SWIR windows only if we have wavelength axis
    if wavelengths is not None and swir_windows is not None:
        for (lo, hi) in swir_windows:
            plt.axvspan(lo, hi, color="gray", alpha=0.15)

    # Axis labels and title
    if wavelengths is not None:
        plt.xlabel("Wavelength (nm)", fontsize=11)
    else:
        plt.xlabel("Spectral band index", fontsize=11)
    plt.ylabel("Signal amplitude (a.u.)", fontsize=11)
    plt.title(title, fontsize=13)

    # Grid, legend, layout
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"✓ Saved figure: {out_png}")

def main():
    # -------- Full dataset --------
    X, y = _load_xy(X_FILE, Y_FILE)
    n_bands = X.shape[1]
    wavelengths = _maybe_load_wavelengths(n_bands)
    x_axis = wavelengths if wavelengths is not None else np.arange(n_bands)

    mu0_full, mu1_full = _class_means(X, y)
    _plot_means(
        out_png=os.path.join(DATA_PATH, "mean_spectra_highlight_full.png"),
        title="Class-average spectra (Full dataset) with SWIR windows",
        x_axis=x_axis,
        mu0=mu0_full,
        mu1=mu1_full,
        wavelengths=wavelengths,
        swir_windows=SWIR_WINDOWS
    )
    _save_csv_means(
        path_csv=os.path.join(DATA_PATH, "mean_spectra_full.csv"),
        x_axis=x_axis,
        mu0=mu0_full,
        mu1=mu1_full,
        wavelengths=wavelengths
    )

    # -------- Selected dataset (optional) --------
    if os.path.exists(XSEL_FILE):
        Xs = np.load(XSEL_FILE)
        if Xs.ndim == 3 and Xs.shape[-1] == 1:
            Xs = Xs.squeeze(-1)
        if Xs.ndim != 2:
            print(f"[WARN] X_selected.npy has unexpected shape {Xs.shape}; skipping selected plot.")
            return

        # Note: the selected set is already reduced in bands; we will plot by band index.
        mu0_sel, mu1_sel = _class_means(Xs, y)
        x_axis_sel = np.arange(Xs.shape[1])

        _plot_means(
            out_png=os.path.join(DATA_PATH, "mean_spectra_highlight_selected.png"),
            title="Class-average spectra (Selected bands) — index domain",
            x_axis=x_axis_sel,
            mu0=mu0_sel,
            mu1=mu1_sel,
            wavelengths=None,            # no nm axis for reduced set (unless you map indices back to wavelengths)
            swir_windows=None            # cannot shade by nm without mapping
        )
        _save_csv_means(
            path_csv=os.path.join(DATA_PATH, "mean_spectra_selected.csv"),
            x_axis=x_axis_sel,
            mu0=mu0_sel,
            mu1=mu1_sel,
            wavelengths=None
        )
    else:
        print("[INFO] X_selected.npy not found; skipping Selected plot.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot ANOVA F-scores by spectral band for HSI-SWIR dataset, with optional wavelength shading
and selected-band markers.

Expected inputs in ./hsi_cnn:
  - X.npy (n_samples, n_bands)                 # required
  - y.npy (n_samples,)                          # required (0 = non-ATCC, 1 = ATCC)
  - wavelengths.npy (n_bands,)                  # optional, nm
  - selected_indices.npy OR idx_selected.npy    # optional, indices of selected bands

Outputs:
  - ./hsi_cnn/anova_f_scores.csv
  - ./hsi_cnn/anova_f_scores.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.feature_selection import f_classif

# ---------------------------
# Configuration
# ---------------------------
DATA_PATH = "./hsi_cnn"
X_FILE = os.path.join(DATA_PATH, "X.npy")
Y_FILE = os.path.join(DATA_PATH, "y.npy")
WL_FILE = os.path.join(DATA_PATH, "wavelengths.npy")  # optional
SEL_FILES = [
    os.path.join(DATA_PATH, "selected_indices.npy"),
    os.path.join(DATA_PATH, "idx_selected.npy"),
]

# Biochemical SWIR windows (nm) to highlight
COLORED_WINDOWS = [
    {"range": (1200, 1800), "label": "OH/CH overtones (1200–1800 nm)", "color": "#FFCC66"},
    {"range": (2100, 2300), "label": "C=O combination bands (2100–2300 nm)", "color": "#99CCFF"},
]
SHADE_ALPHA = 0.18


def load_xy(xfile, yfile):
    if not (os.path.exists(xfile) and os.path.exists(yfile)):
        raise FileNotFoundError(f"Missing input files: {xfile} or {yfile}")
    X = np.load(xfile)
    y = np.load(yfile)
    if X.ndim == 3 and X.shape[-1] == 1:
        X = X.squeeze(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_bands). Got {X.shape}")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError(f"y must be 1D with length == X.shape[0]. Got {y.shape} vs {X.shape[0]}")
    return X, y


def maybe_load_wavelengths(n_bands):
    if os.path.exists(WL_FILE):
        wl = np.load(WL_FILE)
        if wl.ndim == 1 and len(wl) == n_bands:
            return wl.astype(float)
        else:
            print(f"[WARN] wavelengths.npy has shape {wl.shape} but expected ({n_bands},). Ignoring.")
    return None


def maybe_load_selected_indices(n_bands):
    for f in SEL_FILES:
        if os.path.exists(f):
            idx = np.load(f)
            idx = np.asarray(idx).astype(int)
            # sanity check
            idx = idx[(idx >= 0) & (idx < n_bands)]
            if idx.size > 0:
                print(f"[INFO] Loaded selected-band indices from {os.path.basename(f)} (n={idx.size}).")
                return np.unique(idx)
    print("[INFO] No selected-band indices found; skipping markers.")
    return None


def compute_anova_fscores(X, y, normalize=True):
    """
    Defensive computation of ANOVA F-scores:
      - replace NaN/Inf
      - if no finite values, fall back to ones to avoid division-by-zero
      - normalize to [0,1] if requested
    """
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    f_vals, _ = f_classif(Xc, y)  # shape (n_bands,)

    # Replace NaN with 0
    f_vals = np.nan_to_num(f_vals, nan=0.0)

    # Handle ±Inf robustly
    finite_vals = f_vals[np.isfinite(f_vals)]
    if finite_vals.size == 0:
        # degenerate case: set to ones to avoid zero division later
        f_vals = np.ones_like(f_vals)
        finite_vals = f_vals

    # Replace +Inf with max finite, -Inf with 0
    max_finite = np.max(finite_vals)
    f_vals[np.isposinf(f_vals)] = max_finite
    f_vals[np.isneginf(f_vals)] = 0.0

    if normalize:
        maxv = np.max(finite_vals)
        if maxv <= 0:
            maxv = 1.0
        f_norm = f_vals / maxv
        return f_vals, f_norm
    return f_vals, f_vals


def save_csv(path_csv, f_raw, f_norm, wavelengths=None):
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write("band_index,wavelength_nm,anova_fscore,anova_fscore_normalized\n")
        for i, (fr, fn) in enumerate(zip(f_raw, f_norm)):
            wl = wavelengths[i] if wavelengths is not None else np.nan
            f.write(f"{i},{wl},{fr},{fn}\n")
    print(f"✓ Saved CSV: {path_csv}")


def plot_anova(out_png, x_axis, f_norm, wavelengths=None, selected_idx=None):
    plt.figure(figsize=(12, 5))

    legend_patches = []
    # Shade biochemical windows if we have wavelengths
    if wavelengths is not None:
        for win in COLORED_WINDOWS:
            lo, hi = win["range"]
            plt.axvspan(lo, hi, color=win["color"], alpha=SHADE_ALPHA, linewidth=0)
            legend_patches.append(Patch(facecolor=win["color"], edgecolor="none",
                                        alpha=SHADE_ALPHA, label=win["label"]))

    # Plot F-score curve
    plt.plot(x_axis, f_norm, linewidth=2, label="ANOVA F-score (normalized)")

    # Mark selected bands if provided
    if selected_idx is not None and selected_idx.size > 0:
        if wavelengths is not None:
            x_sel = wavelengths[selected_idx]
        else:
            x_sel = selected_idx
        y_sel = f_norm[selected_idx]
        plt.scatter(x_sel, y_sel, s=26, marker="o", facecolors="none", edgecolors="crimson",
                    linewidths=1.2, label="Selected bands")

    # Labels, grid, legends
    if wavelengths is not None:
        plt.xlabel("Wavelength (nm)", fontsize=11)
    else:
        plt.xlabel("Spectral band index", fontsize=11)
    plt.ylabel("ANOVA F-score (normalized)", fontsize=11)
    plt.title("Distribution of ANOVA F-scores by spectral band", fontsize=13)

    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    # Compose legend: line + (optional) windows + (optional) selected markers
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
    # Load data
    X, y = load_xy(X_FILE, Y_FILE)
    n_bands = X.shape[1]

    # Optional metadata
    wavelengths = maybe_load_wavelengths(n_bands)
    selected_idx = maybe_load_selected_indices(n_bands)

    # Compute ANOVA F-scores
    f_raw, f_norm = compute_anova_fscores(X, y, normalize=True)

    # Save CSV
    save_csv(os.path.join(DATA_PATH, "anova_f_scores.csv"),
             f_raw=f_raw, f_norm=f_norm, wavelengths=wavelengths)

    # Build x-axis
    x_axis = wavelengths if wavelengths is not None else np.arange(n_bands)

    # Plot
    plot_anova(out_png=os.path.join(DATA_PATH, "anova_f_scores.png"),
               x_axis=x_axis, f_norm=f_norm,
               wavelengths=wavelengths, selected_idx=selected_idx)


if __name__ == "__main__":
    main()

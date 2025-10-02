#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 14:44:19 2025

@author: clarimar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate ANOVA F-score and Random Forest feature-importance figures
with grids and English labels/titles. Optionally uses wavelengths (nm)
if './hsi_cnn/wavelengths.npy' exists; otherwise plots by band index.

Outputs:
- ./hsi_cnn/anova_f_scores.png
- ./hsi_cnn/anova_f_scores.csv
- ./hsi_cnn/rf_band_importance.png          (all bands, normalized)
- ./hsi_cnn/rf_band_importance_top50.png    (top-50 bars, normalized + labels)
- ./hsi_cnn/rf_importances.csv
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------
# Configuration
# -----------------------
DATA_PATH = "./hsi_cnn"
X_FILE = os.path.join(DATA_PATH, "X.npy")
Y_FILE = os.path.join(DATA_PATH, "y.npy")
WL_FILE = os.path.join(DATA_PATH, "wavelengths.npy")  # optional

# Biochemically relevant SWIR windows (approx., in nm)
BIO_WINDOWS = [(1200, 1800), (2100, 2300)]

# RF settings
N_ESTIMATORS = 200
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
TOP_K = 50

# -----------------------
# Load data
# -----------------------
if not os.path.exists(X_FILE) or not os.path.exists(Y_FILE):
    raise FileNotFoundError(
        f"Could not find X/y at {X_FILE} / {Y_FILE}. "
        "Please ensure the dataset is saved under ./hsi_cnn."
    )

X = np.load(X_FILE)
y = np.load(Y_FILE)

# Ensure (n_samples, n_bands)
if X.ndim == 3 and X.shape[-1] == 1:
    X = X.squeeze(-1)
if X.ndim != 2:
    raise ValueError(f"X must be 2D (n_samples, n_bands). Got shape: {X.shape}")

n_samples, n_bands = X.shape

# Try to load wavelengths (nm) if available
wavelengths = None
if os.path.exists(WL_FILE):
    wl = np.load(WL_FILE)
    if wl.ndim == 1 and len(wl) == n_bands:
        wavelengths = wl.astype(float)
    else:
        print(f"[WARN] wavelengths.npy has shape {wl.shape}, expected ({n_bands},). Ignoring.")
x_axis = wavelengths if wavelengths is not None else np.arange(n_bands)
x_label = "Wavelength (nm)" if wavelengths is not None else "Spectral band index"

# -----------------------
# ANOVA F-scores (per band)
# -----------------------
# f_values can contain inf/nan if a band is constant within classes; sanitize
f_values, _ = f_classif(X, y)
# Replace NaN/Inf with 0 for plotting, then normalize to [0,1]
f_values = np.nan_to_num(f_values, nan=0.0, posinf=np.nanmax(f_values[np.isfinite(f_values)]) if np.isfinite(f_values).any() else 0.0, neginf=0.0)
maxf = np.max(f_values) if np.max(f_values) > 0 else 1.0
f_norm = f_values / maxf

# Save CSV for reproducibility
np.savetxt(os.path.join(DATA_PATH, "anova_f_scores.csv"),
           np.c_[np.arange(n_bands), (wavelengths if wavelengths is not None else np.full(n_bands, np.nan)), f_values, f_norm],
           delimiter=",",
           header="band_index,wavelength_nm,f_value,f_value_normalized",
           comments="")

plt.figure(figsize=(12, 5))
plt.plot(x_axis, f_norm, linewidth=2, label="ANOVA F-score (normalized)")
for (lo, hi) in BIO_WINDOWS:
    if wavelengths is not None:
        # Shade by nm directly
        plt.axvspan(lo, hi, color="gray", alpha=0.15, label=None)
    else:
        # Shade by approximate band indices if user only has indices
        # (Best-effort: skip shading if wavelengths are unknown.)
        pass

plt.title("Distribution of ANOVA F-scores by spectral band", fontsize=13)
plt.xlabel(x_label, fontsize=11)
plt.ylabel("F-score (normalized)", fontsize=11)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, "anova_f_scores.png"), dpi=200)
plt.close()

print("✓ Saved ANOVA figure: ./hsi_cnn/anova_f_scores.png")
print("✓ Saved ANOVA CSV   : ./hsi_cnn/anova_f_scores.csv")

# -----------------------
# Random Forest importance
# -----------------------
# Scale features for robustness (RF doesn't require scaling, but can help stability in some cases)
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    class_weight=CLASS_WEIGHT,
    n_jobs=-1
)
rf.fit(X_scaled, y)

importances = rf.feature_importances_.astype(float)
# Normalize to [0,1] for plotting
imax = importances.max() if importances.max() > 0 else 1.0
imp_norm = importances / imax

# Save CSV
np.savetxt(os.path.join(DATA_PATH, "rf_importances.csv"),
           np.c_[np.arange(n_bands), (wavelengths if wavelengths is not None else np.full(n_bands, np.nan)), importances, imp_norm],
           delimiter=",",
           header="band_index,wavelength_nm,rf_importance,rf_importance_normalized",
           comments="")

# (1) All bands (line)
plt.figure(figsize=(12, 5))
plt.plot(x_axis, imp_norm, linewidth=2, label="RF importance (normalized)")
for (lo, hi) in BIO_WINDOWS:
    if wavelengths is not None:
        plt.axvspan(lo, hi, color="gray", alpha=0.15)
plt.title("Random Forest importance across spectral bands (normalized)", fontsize=13)
plt.xlabel(x_label, fontsize=11)
plt.ylabel("Importance (normalized)", fontsize=11)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, "rf_band_importance.png"), dpi=200)
plt.close()

# (2) Top-K bars
top_idx = np.argsort(importances)[::-1][:TOP_K]
top_imp = importances[top_idx]
top_imp_norm = top_imp / (imax if imax > 0 else 1.0)

# Sort the top-K by wavelength or by importance for nicer plotting
# Here we sort by importance descending:
order = np.argsort(top_imp_norm)[::-1]
top_idx_sorted = top_idx[order]
top_imp_norm_sorted = top_imp_norm[order]

plt.figure(figsize=(12, 6))
plt.bar(np.arange(TOP_K), top_imp_norm_sorted, width=0.8)
plt.title(f"Top {TOP_K} spectral bands by Random Forest importance (normalized)", fontsize=13)
plt.xlabel("Ranked bands (descending importance)", fontsize=11)
plt.ylabel("Importance (normalized)", fontsize=11)
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
# Annotate a few top bars (optional: top 10)
for i in range(min(10, TOP_K)):
    bidx = top_idx_sorted[i]
    wl_txt = f"{wavelengths[bidx]:.0f} nm" if wavelengths is not None else f"Band {bidx}"
    plt.text(i, top_imp_norm_sorted[i] + 0.01, wl_txt, rotation=90, ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, "rf_band_importance_top50.png"), dpi=200)
plt.close()

print("✓ Saved RF figures :")
print("   - ./hsi_cnn/rf_band_importance.png (all bands, normalized)")
print("   - ./hsi_cnn/rf_band_importance_top50.png (top-50 bars, normalized)")
print("✓ Saved RF CSV     : ./hsi_cnn/rf_importances.csv")

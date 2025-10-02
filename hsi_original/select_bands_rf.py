#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Band Selection using Random Forest Importance
Author: Clarimar
Date: Aug 11, 2025
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =============================
# 📂 Directories
# =============================
DATA_PATH = './hsi_cnn/'

# =============================
# 🚚 Load Data
# =============================
print("📥 Loading data...")
X = np.load(os.path.join(DATA_PATH, 'X.npy'))
y = np.load(os.path.join(DATA_PATH, 'y.npy'))

print(f"✅ Data loaded: X shape {X.shape}, y shape {y.shape}")

# =============================
# 🚀 Preprocessing
# =============================
# Remove last dimension if data has shape (N, Bands, 1)
if X.ndim == 3 and X.shape[-1] == 1:
    X = X.squeeze(-1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# 🌳 Random Forest Importance
# =============================
print("🚀 Training Random Forest to compute band importance...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_scaled, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# =============================
# 🔥 Select Top N Bands
# =============================
N_BANDS = 100  # Adjust as needed
top_indices = indices[:N_BANDS]
X_selected = X_scaled[:, top_indices]

print(f"✅ Selected bands (indices): {top_indices}")

# =============================
# 📊 Plot Band Importances
# =============================
# Full importance plot
plt.figure(figsize=(12, 6))
plt.title("Band Importance - Random Forest")
plt.bar(range(len(importances)), importances[indices])
plt.xlabel("Band Index (sorted by importance)")
plt.ylabel("Importance (Gini)")
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'rf_band_importance.png'), dpi=150)
plt.close()

# Top N bands plot
plt.figure(figsize=(10, 5))
plt.title(f"Top {N_BANDS} Selected Bands - Random Forest")
plt.bar(range(N_BANDS), importances[top_indices])
plt.xlabel("Rank")
plt.ylabel("Importance (Gini)")
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'rf_selected_band_importance.png'), dpi=150)
plt.close()

print("📈 Plots saved: rf_band_importance.png and rf_selected_band_importance.png")

# =============================
# 💾 Save Selected Data
# =============================
np.save(os.path.join(DATA_PATH, 'X_selected.npy'), X_selected)
np.save(os.path.join(DATA_PATH, 'y_selected.npy'), y)

print(f"💾 Data saved: X_selected.npy (shape {X_selected.shape}), y_selected.npy (shape {y.shape})")
print("🚀 Band selection completed successfully.")

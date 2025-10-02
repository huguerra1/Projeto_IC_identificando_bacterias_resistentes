#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:17:53 2025

@author: clarimar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
# (opcional para servidores sem GUI)
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# ==========================
# PATHS
# ==========================
DATA_PATH = './hsi_cnn/'
os.makedirs(DATA_PATH, exist_ok=True)

# ==========================
# LOAD DATA
# ==========================
X = np.load(os.path.join(DATA_PATH, 'X.npy'))   # shape: (N_samples, N_bands)
y = np.load(os.path.join(DATA_PATH, 'y.npy'))   # shape: (N_samples,)

# sanity checks
if np.isnan(X).any():
    raise ValueError("Data contains NaN values.")

num_bands = X.shape[1]

# ==========================
# ANOVA F-VALUE (normalized)
# ==========================
f_values, _ = f_classif(X, y)
f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
f_values /= max(np.max(f_values), 1e-12)

# ==========================
# RANDOM FOREST IMPORTANCE (normalized)
# ==========================
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_importance = rf.feature_importances_
rf_importance = np.nan_to_num(rf_importance, nan=0.0, posinf=0.0, neginf=0.0)
rf_importance /= max(np.max(rf_importance), 1e-12)

# ==========================
# SIMPLE VIP PROXY (between-class difference, normalized)
# ==========================
class1_mean = np.mean(X[y == 1], axis=0) if np.any(y == 1) else np.zeros(num_bands)
class0_mean = np.mean(X[y == 0], axis=0) if np.any(y == 0) else np.zeros(num_bands)
group_mean_diff = np.abs(class1_mean - class0_mean)
group_mean_diff = np.nan_to_num(group_mean_diff, nan=0.0, posinf=0.0, neginf=0.0)
vip_scores = group_mean_diff / max(np.max(group_mean_diff), 1e-12)

# ==========================
# PLOT — EN + GRID
# ==========================
plt.figure(figsize=(12, 6))
plt.plot(f_values, label='ANOVA F-value')
plt.plot(rf_importance, label='Random Forest Importance')
plt.plot(vip_scores, label='VIP Proxy (between-class diff)')
plt.title('Comparison of Band-Selection Methods')
plt.xlabel('Band Index')
plt.ylabel('Normalized Score')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
plt.tight_layout()
out_png = os.path.join(DATA_PATH, 'band_selection_analysis.png')
plt.savefig(out_png, dpi=150)
# plt.show()  # habilite se estiver rodando com GUI
plt.close()

# ==========================
# SAVE SCORES (CSV)
# ==========================
df = pd.DataFrame({
    'band_index': np.arange(num_bands, dtype=int),
    'anova_fvalue_norm': f_values,
    'rf_importance_norm': rf_importance,
    'vip_proxy_norm': vip_scores
})
out_csv = os.path.join(DATA_PATH, 'band_selection_scores.csv')
df.to_csv(out_csv, index=False)

print("✅ Analysis complete.")
print(f" - Figure saved to: {out_png}")
print(f" - Scores saved to: {out_csv}")

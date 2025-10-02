#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # headless-friendly
import matplotlib.pyplot as plt

# ===========================
# PATHS & CONFIG
# ===========================
DATA_PATH = './hsi_cnn/'
os.makedirs(DATA_PATH, exist_ok=True)

# Choose the model: 'full' or 'selected'
model_type = 'full'   # change to 'selected' to use the selected-bands model
THRESHOLD = 0.5       # decision threshold

if model_type == 'full':
    model_file = os.path.join(DATA_PATH, 'cnn1d_model_full.keras')
    data_file  = os.path.join(DATA_PATH, 'X.npy')
    name_tag   = 'full'
elif model_type == 'selected':
    model_file = os.path.join(DATA_PATH, 'cnn1d_model_selected.keras')
    data_file  = os.path.join(DATA_PATH, 'X_selected.npy')
    name_tag   = 'selected'
else:
    raise ValueError("model_type must be 'full' or 'selected'.")

# ===========================
# LOAD MODEL
# ===========================
print(f"📥 Loading model: {model_file}")
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")
model = tf.keras.models.load_model(model_file)
print("✅ Model loaded.")

# ===========================
# LOAD DATA TO PREDICT
# ===========================
print(f"📥 Loading data: {data_file}")
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Input data file not found: {data_file}")

X_new = np.load(data_file)
# ensure channel dimension for 1D-CNN input
if X_new.ndim == 2:
    X_new = X_new[..., np.newaxis]
elif X_new.ndim != 3:
    raise ValueError(f"Expected X_new with 2D or 3D shape, got {X_new.shape}")

print(f"🔸 Data for prediction: {X_new.shape}")

# ===========================
# PREDICT
# ===========================
print("🚀 Running predictions...")
y_pred_prob = model.predict(X_new, verbose=0).ravel()
y_pred = (y_pred_prob >= THRESHOLD).astype(int)

# ===========================
# SAVE RESULTS (CSV + indices)
# ===========================
results = pd.DataFrame({
    'Predicted_Probability_Positive': y_pred_prob,
    'Predicted_Class': y_pred
})
result_file = os.path.join(DATA_PATH, f'predictions_{name_tag}.csv')
results.to_csv(result_file, index=False)
print(f"\n✅ Predictions saved to: {result_file}")

# Save indices for convenience
pos_idx = np.where(y_pred == 1)[0]
neg_idx = np.where(y_pred == 0)[0]
np.save(os.path.join(DATA_PATH, f'pred_positive_indices_{name_tag}.npy'), pos_idx)
np.save(os.path.join(DATA_PATH, f'pred_negative_indices_{name_tag}.npy'), neg_idx)
print(f"🔖 Positive indices: {pos_idx.size} | Negative indices: {neg_idx.size}")

# ===========================
# PLOTS (all with grid)
# ===========================

# 1) Probability histogram
plt.figure(figsize=(7, 4))
plt.hist(y_pred_prob, bins=40, alpha=0.9)
plt.axvline(THRESHOLD, linestyle='--', linewidth=1.2, label=f'Threshold = {THRESHOLD:.2f}')
plt.title(f'Prediction Probability Histogram — {name_tag.capitalize()}')
plt.xlabel('Predicted Probability (Positive Class)')
plt.ylabel('Count')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
hist_file = os.path.join(DATA_PATH, f'pred_prob_hist_{name_tag}.png')
plt.savefig(hist_file, dpi=150)
plt.close()
print(f"🖼️ Histogram saved to: {hist_file}")

# 2) ECDF (Empirical Cumulative Distribution Function)
sorted_prob = np.sort(y_pred_prob)
ecdf = np.arange(1, sorted_prob.size + 1) / sorted_prob.size
plt.figure(figsize=(7, 4))
plt.plot(sorted_prob, ecdf, drawstyle='steps-post', label='ECDF')
plt.axvline(THRESHOLD, linestyle='--', linewidth=1.2, label=f'Threshold = {THRESHOLD:.2f}')
plt.title(f'ECDF of Predicted Probabilities — {name_tag.capitalize()}')
plt.xlabel('Predicted Probability (Positive Class)')
plt.ylabel('Cumulative Fraction')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
ecdf_file = os.path.join(DATA_PATH, f'pred_prob_ecdf_{name_tag}.png')
plt.savefig(ecdf_file, dpi=150)
plt.close()
print(f"🖼️ ECDF saved to: {ecdf_file}")

# 3) Boxplot of probabilities
plt.figure(figsize=(6, 4))
plt.boxplot(y_pred_prob, vert=True, labels=[f'{name_tag.capitalize()}'])
plt.title(f'Predicted Probability Boxplot — {name_tag.capitalize()}')
plt.ylabel('Predicted Probability (Positive Class)')
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
box_file = os.path.join(DATA_PATH, f'pred_prob_boxplot_{name_tag}.png')
plt.savefig(box_file, dpi=150)
plt.close()
print(f"🖼️ Boxplot saved to: {box_file}")

# ===========================
# QUICK SUMMARY
# ===========================
q = np.quantile(y_pred_prob, [0.05, 0.25, 0.5, 0.75, 0.95])
print("\n🔎 Probability summary (quantiles):")
print(f"  5%:  {q[0]:.3f}")
print(f" 25%:  {q[1]:.3f}")
print(f" 50%:  {q[2]:.3f}")
print(f" 75%:  {q[3]:.3f}")
print(f" 95%:  {q[4]:.3f}")
print(f"  Mean: {np.mean(y_pred_prob):.3f} | Std: {np.std(y_pred_prob):.3f}")
print(f"  Positives @ {THRESHOLD:.2f}: {pos_idx.size} / {y_pred_prob.size} ({100*pos_idx.size/y_pred_prob.size:.1f}%)")

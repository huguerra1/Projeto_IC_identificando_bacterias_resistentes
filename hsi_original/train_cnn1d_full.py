#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================
# 🚀 1D-CNN Training — Full Model (all bands)
# =============================

import os
import json
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # headless-friendly
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter

# =============================
# 📂 Data directory
# =============================
DATA_PATH = './hsi_cnn/'
os.makedirs(DATA_PATH, exist_ok=True)

# =============================
# 🚚 Load data
# =============================
print("📥 Loading data...")
X = np.load(os.path.join(DATA_PATH, 'X.npy'))
y = np.load(os.path.join(DATA_PATH, 'y.npy'))
print(f"✅ Data loaded: X shape {X.shape}, y shape {y.shape}")

# Add channel dimension for 1D CNN
if X.ndim == 2:
    X = X[..., np.newaxis]  # (samples, bands, 1)

# =============================
# 🔀 Split data
# =============================
print("📊 Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print("✅ Split done:")
print(f"🔹 Train:      {X_train.shape}")
print(f"🔹 Validation: {X_val.shape}")
print(f"🔹 Test:       {X_test.shape}")

# =============================
# 🧠 Define 1D-CNN model
# =============================
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =============================
# 🔥 Callbacks
# =============================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    os.path.join(DATA_PATH, 'model_cnn1d_full.keras'),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# =============================
# ⚖️ Class weights (imbalance handling)
# =============================
counter = Counter(y_train)
total = sum(counter.values())
class_weight = {0: total / counter[0], 1: total / counter[1]}
print(f"📊 Class Weights: {class_weight}")

# =============================
# 🚀 Training
# =============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=512,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weight,
    verbose=2
)

# =============================
# 💾 Save final model (last epoch)
# =============================
final_model_path = os.path.join(DATA_PATH, 'model_cnn1d_full_final.keras')
model.save(final_model_path)
print(f"✅ Final model saved to {final_model_path}")

# =============================
# 💾 Save training history (PKL, JSON, CSV)
# =============================
hist_dict = history.history
with open(os.path.join(DATA_PATH, 'history_full.pkl'), 'wb') as f:
    pickle.dump(hist_dict, f)
with open(os.path.join(DATA_PATH, 'history_full.json'), 'w') as f:
    json.dump(hist_dict, f, indent=2)
pd.DataFrame(hist_dict).to_csv(os.path.join(DATA_PATH, 'history_full.csv'), index=False)
print("🧾 History saved as history_full.pkl / .json / .csv")

# =============================
# 📈 Plot loss & accuracy curves (EN + grid)
# =============================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_dict['loss'], label='Train Loss')
plt.plot(hist_dict['val_loss'], label='Validation Loss')
plt.title('Loss Curve — 1D-CNN (Full Spectrum)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)

plt.subplot(1, 2, 2)
plt.plot(hist_dict['accuracy'], label='Train Accuracy')
plt.plot(hist_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve — 1D-CNN (Full Spectrum)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
la_path = os.path.join(DATA_PATH, "loss_accuracy_full.png")
plt.savefig(la_path, dpi=150)
plt.close()
print(f"📈 Curves saved as {la_path}")

# =============================
# 🧠 Evaluation on the test set
# =============================
print("🚀 Evaluating on the test set...")
y_pred_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

report = classification_report(y_test, y_pred, target_names=["Staph spp.", "ATCC"])
print(report)

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix (counts):\n{cm}")

# =============================
# 📊 Confusion matrices (counts and row-normalized %)
# =============================
classes = ['Staph spp.', 'ATCC']

# Counts
plt.figure(figsize=(6, 5))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
ax.set_title('Confusion Matrix (Counts)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_xticklabels(classes)
ax.set_yticklabels(classes, rotation=0)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
cm_counts_path = os.path.join(DATA_PATH, 'confusion_matrix_full_counts.png')
plt.savefig(cm_counts_path, dpi=150)
plt.close()

# Row-normalized (%)
row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
cm_norm_pct = 100.0 * (cm.astype(np.float64) / row_sums)
annot = np.empty_like(cm_norm_pct, dtype=object)
for i in range(cm_norm_pct.shape[0]):
    for j in range(cm_norm_pct.shape[1]):
        annot[i, j] = f"{cm_norm_pct[i, j]:.1f}%"

plt.figure(figsize=(6, 5))
ax = sns.heatmap(
    cm_norm_pct, annot=annot, fmt='',
    cmap='Blues', cbar=True, vmin=0, vmax=100,
    cbar_kws={'format': '%.0f%%'}
)
ax.set_title('Confusion Matrix (Row-normalized, %)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_xticklabels(classes)
ax.set_yticklabels(classes, rotation=0)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
cm_norm_path = os.path.join(DATA_PATH, 'confusion_matrix_full_normalized_pct.png')
plt.savefig(cm_norm_path, dpi=150)
plt.close()

# =============================
# 📊 ROC curve (EN + grid)
# =============================
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — 1D-CNN (Full Spectrum)')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
roc_path = os.path.join(DATA_PATH, 'roc_curve_full.png')
plt.savefig(roc_path, dpi=150)
plt.close()

print("\n✅ Saved figures:")
print(f" - {la_path}")
print(f" - {cm_counts_path}")
print(f" - {cm_norm_path}")
print(f" - {roc_path}")

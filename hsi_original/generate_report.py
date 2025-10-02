#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)
from fpdf import FPDF

# =============================
# CONFIGURATION
# =============================
DATA_PATH = './hsi_cnn/'
MODEL_PATH = './'  # adjust if needed
MODE = 'full'  # 'full' or 'selected'

if MODE == 'full':
    X_file = 'X.npy'
    model_file = 'model_cnn1d_full.keras'
    report_name = 'report_full.pdf'
else:
    X_file = 'X_selected.npy'
    model_file = 'model_cnn1d_selected.keras'
    report_name = 'report_selected.pdf'

# =============================
# LOAD DATA
# =============================
X = np.load(os.path.join(DATA_PATH, X_file))
y = np.load(os.path.join(DATA_PATH, 'y.npy'))

# Ensure 3D input for 1D CNN
if X.ndim == 2:
    X = X[..., np.newaxis]

print(f"✅ Data loaded: X shape {X.shape}, y shape {y.shape}")

# =============================
# LOAD MODEL
# =============================
model = tf.keras.models.load_model(os.path.join(MODEL_PATH, model_file))
print(f"✅ Model loaded: {model_file}")

# =============================
# TRAIN/VAL/TEST SPLIT (recreate test subset)
# =============================
from sklearn.model_selection import train_test_split

_, X_temp, _, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# =============================
# PREDICTIONS
# =============================
y_pred_proba = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

# =============================
# CLASSIFICATION REPORT
# =============================
report_text = classification_report(
    y_test, y_pred, target_names=["Staph spp.", "ATCC"]
)
print(report_text)

# =============================
# CONFUSION MATRIX (with grid)
# =============================
cm = confusion_matrix(y_test, y_pred)
classes = ["Staph spp.", "ATCC"]

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Count', rotation=90)

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, ha='right')
plt.yticks(tick_marks, classes)

# cell annotations
thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

# grid overlay (cell borders)
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8)
ax.tick_params(which='minor', bottom=False, left=False)

# generic dashed grid as well (optional aesthetic)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_path = f'confusion_matrix_{MODE}.png'
plt.savefig(cm_path, dpi=150)
plt.close()

# =============================
# ROC CURVE (with grid)
# =============================
from sklearn.metrics import auc
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
roc_path = f'roc_curve_{MODE}.png'
plt.savefig(roc_path, dpi=150)
plt.close()

print(f"✔️ AUC: {roc_auc:.4f}")

# =============================
# PDF REPORT (EN)
# =============================
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, f'CNN 1D Report - {MODE.upper()}', ln=True, align='C')

pdf.set_font('Arial', '', 12)
pdf.ln(8)
pdf.multi_cell(0, 8, f'ATCC vs Staphylococcus spp. classification\nAUC: {roc_auc:.4f}')
pdf.ln(3)

pdf.set_font('Courier', size=10)
pdf.multi_cell(0, 5, report_text)

# Confusion Matrix page
pdf.add_page()
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Confusion Matrix', ln=True, align='C')
if os.path.exists(cm_path):
    pdf.image(cm_path, x=20, y=30, w=170)

# ROC Curve page
pdf.add_page()
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'ROC Curve', ln=True, align='C')
if os.path.exists(roc_path):
    pdf.image(roc_path, x=20, y=30, w=170)

# Optional: add loss/accuracy page if figure exists
loss_file = f'loss_accuracy_{MODE}.png'
if os.path.exists(loss_file):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Loss and Accuracy Curves', ln=True, align='C')
    pdf.image(loss_file, x=20, y=30, w=170)

pdf.output(report_name)
print(f"✅ PDF report generated: {report_name}")

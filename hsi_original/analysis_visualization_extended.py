#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analysis_visualization_extended.py

Avaliação completa para os modelos Full e Selected, com artefatos salvos em
./output_analysis/full e ./output_analysis/selected.

Inclui (por modelo):
- metrics_<prefix>.csv / .json (Accuracy, Precision, Recall, F1, ROC_AUC, Average_Precision)
- classification_report_<prefix>.txt
- confusion_matrix_<prefix>_counts.png / .csv
- confusion_matrix_<prefix>_normalized_pct.png / .csv
- roc_curve_<prefix>.png + roc_points_<prefix>.csv (fpr, tpr, threshold)
- precision_recall_<prefix>.png + precision_recall_<prefix>.csv (recall, precision, threshold)
- loss_accuracy_<prefix>.png (se history_*.pkl existir)
- error_spectra_<prefix>.csv + .png (FP/FN médios, se existirem)
- PCA (csvs + plot)
- ***NOVO*** ANOVA F-scores por banda (csv + plot com regiões funcionais opcionais)
- ***NOVO*** RF importances normalizadas (csv + plot + Top-50 csv)
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier

# =============================
# Paths & setup
# =============================
DATA_PATH = "./hsi_cnn"
OUT_PATH = "./output_analysis"
os.makedirs(OUT_PATH, exist_ok=True)

CONFIGS = [
    {
        "name": "full",
        "label": "Full",
        "x_file": "X.npy",
        "y_file": "y.npy",
        "model_file": "model_cnn1d_full.keras",
        "history_pkl": "history_full.pkl",
    },
    {
        "name": "selected",
        "label": "Selected",
        "x_file": "X_selected.npy",
        "y_file": "y.npy",
        "model_file": "model_cnn1d_selected.keras",
        "history_pkl": "history_selected.pkl",
    },
]

RANDOM_STATE = 42
TEST_SIZE = 0.3
CLASS_NAMES = ["Staph spp.", "ATCC"]

# Limite de pontos para PCA (para não estourar memória/tempo)
PCA_MAX_POINTS = 50000
PCA_TOP_LOADINGS = 20  # quantos loadings salvar por componente (|loading|)
RF_TOP_K = 50          # quantas bandas destacar no gráfico/CSV do RF

# =============================
# Utils
# =============================
def safe_load_history(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                hist = pickle.load(f)
            if isinstance(hist, dict) and (("loss" in hist) or ("accuracy" in hist)):
                return hist
        except Exception as e:
            print(f"⚠️ Could not load history from {path}: {e}")
    return None


def ensure_channel_dim(X):
    if X.ndim == 2:
        return X[..., np.newaxis]
    return X


def _flatten_features(X):
    """(n, bands, 1) -> (n, bands)"""
    if X.ndim == 3 and X.shape[-1] == 1:
        return X.squeeze(-1)
    return X


def _stratified_subsample_idx(y, max_points, rng):
    """Retorna índices estratificados até max_points (ou menos se n<max)."""
    n = y.shape[0]
    if n <= max_points:
        return np.arange(n)
    classes, counts = np.unique(y, return_counts=True)
    props = counts / counts.sum()
    per_class = (props * max_points).astype(int)
    while per_class.sum() < max_points:
        per_class[np.argmax(props - per_class / max_points)] += 1
    idxs = []
    for c, k in zip(classes, per_class):
        cand = np.where(y == c)[0]
        pick = rng.choice(cand, size=min(k, cand.size), replace=False)
        idxs.append(pick)
    return np.concatenate(idxs)


def load_wavelengths_if_any():
    """Tenta carregar comprimentos de onda (nm) de wavelengths.npy ou wavelengths.csv."""
    npy_path = os.path.join(DATA_PATH, "wavelengths.npy")
    csv_path = os.path.join(DATA_PATH, "wavelengths.csv")
    if os.path.exists(npy_path):
        wl = np.load(npy_path)
        return wl.astype(float)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        col = None
        for c in df.columns:
            if c.strip().lower() in {"wavelength_nm", "wavelength", "nm"}:
                col = c
                break
        if col is None and len(df.columns) == 1:
            col = df.columns[0]
        if col is not None:
            return df[col].to_numpy(dtype=float)
    return None


def load_regions_if_any():
    """
    Tenta carregar regiões funcionais (nm) de regions.csv com colunas:
    start_nm,end_nm,label
    """
    csv_path = os.path.join(DATA_PATH, "regions.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        needed = {"start_nm", "end_nm", "label"}
        if needed.issubset(set(map(str.lower, df.columns))):
            # normaliza nomes
            cols = {c: c.lower() for c in df.columns}
            df.rename(columns=cols, inplace=True)
            return df[["start_nm", "end_nm", "label"]].copy()
        else:
            print("⚠️ regions.csv found but missing required columns: start_nm,end_nm,label")
    return None

# =============================
# Plotters & Savers (confusion/ROC/PR/history/error spectra/PCA)
# =============================
def plot_confusions_and_save(cm, classes, prefix, out_dir):
    paths = {}

    # Counts FIG
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    ax.set_title(f'Confusion Matrix (Counts) — {prefix}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes, rotation=0)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    counts_png = os.path.join(out_dir, f'confusion_matrix_{prefix}_counts.png')
    plt.savefig(counts_png, dpi=150)
    plt.close()
    paths["cm_counts_png"] = counts_png

    # Counts CSV
    counts_csv = os.path.join(out_dir, f'confusion_matrix_{prefix}_counts.csv')
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(counts_csv)
    paths["cm_counts_csv"] = counts_csv

    # Row-normalized (%)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm_pct = 100.0 * (cm.astype(np.float64) / row_sums)

    # Normalized FIG (annot as %)
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
    ax.set_title(f'Confusion Matrix (Row-normalized, %) — {prefix}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes, rotation=0)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    norm_png = os.path.join(out_dir, f'confusion_matrix_{prefix}_normalized_pct.png')
    plt.savefig(norm_png, dpi=150)
    plt.close()
    paths["cm_norm_png"] = norm_png

    # Normalized CSV
    norm_csv = os.path.join(out_dir, f'confusion_matrix_{prefix}_normalized_pct.csv')
    pd.DataFrame(cm_norm_pct, index=classes, columns=classes).to_csv(norm_csv)
    paths["cm_norm_csv"] = norm_csv

    return paths


def plot_save_roc(y_true, y_proba, prefix, out_dir):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    roc_csv = os.path.join(out_dir, f"roc_points_{prefix}.csv")
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(roc_csv, index=False)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {prefix}')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    roc_png = os.path.join(out_dir, f'roc_curve_{prefix}.png')
    plt.savefig(roc_png, dpi=150)
    plt.close()

    return roc_png, roc_csv, roc_auc


def plot_save_pr(y_true, y_proba, prefix, out_dir):
    precision, recall, thr = precision_recall_curve(y_true, y_proba)
    thr_full = np.append(thr, np.nan)
    ap = np.trapz(precision[::-1], recall[::-1])

    pr_csv = os.path.join(out_dir, f"precision_recall_{prefix}.csv")
    pd.DataFrame({"recall": recall, "precision": precision, "threshold": thr_full}).to_csv(pr_csv, index=False)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR curve (area ≈ {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {prefix}')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    pr_png = os.path.join(out_dir, f'precision_recall_{prefix}.png')
    plt.savefig(pr_png, dpi=150)
    plt.close()

    return pr_png, pr_csv, ap


def plot_save_history(hist_dict, prefix, out_dir):
    if not hist_dict:
        print(f"ℹ️  No training history available for '{prefix}'.")
        return None

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    if 'loss' in hist_dict: plt.plot(hist_dict['loss'], label='Train Loss')
    if 'val_loss' in hist_dict: plt.plot(hist_dict['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve — {prefix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.subplot(1, 2, 2)
    if 'accuracy' in hist_dict: plt.plot(hist_dict['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in hist_dict: plt.plot(hist_dict['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy Curve — {prefix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, f'loss_accuracy_{prefix}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def compute_and_save_error_spectra(X_test, y_true, y_pred, prefix, out_dir):
    Xb = _flatten_features(X_test)
    if Xb.ndim != 2:
        return None, None

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    if not (fp_mask.any() or fn_mask.any()):
        return None, None

    fp_mean = Xb[fp_mask].mean(axis=0) if fp_mask.any() else None
    fn_mean = Xb[fn_mask].mean(axis=0) if fn_mask.any() else None

    bands = np.arange(Xb.shape[1])
    df = pd.DataFrame({"band_index": bands})
    df["fp_mean"] = fp_mean if fp_mean is not None else np.nan
    df["fn_mean"] = fn_mean if fn_mean is not None else np.nan

    csv_path = os.path.join(out_dir, f"error_spectra_{prefix}.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 4.5))
    if fp_mean is not None:
        plt.plot(bands, fp_mean, label="False Positives — mean spectrum")
    if fn_mean is not None:
        plt.plot(bands, fn_mean, label="False Negatives — mean spectrum")
    plt.title(f'Error Spectra — {prefix}')
    plt.xlabel('Band index')
    plt.ylabel('Amplitude (a.u.)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"error_spectra_{prefix}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    return csv_path, png_path


def pca_analysis_save(X_test, y_true, y_pred, prefix, out_dir):
    rng = np.random.default_rng(RANDOM_STATE)
    Xb = _flatten_features(X_test)
    if Xb.ndim != 2:
        return None

    idx = _stratified_subsample_idx(y_true, PCA_MAX_POINTS, rng)
    X_sub, y_sub, y_pred_sub = Xb[idx], y_true[idx], y_pred[idx]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_sub)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(Xs)

    proj_df = pd.DataFrame({
        "PC1": Z[:, 0],
        "PC2": Z[:, 1],
        "true_label": y_sub,
        "pred_label": y_pred_sub
    })
    cond_tp = (y_sub == 1) & (y_pred_sub == 1)
    cond_tn = (y_sub == 0) & (y_pred_sub == 0)
    cond_fp = (y_sub == 0) & (y_pred_sub == 1)
    cond_fn = (y_sub == 1) & (y_pred_sub == 0)
    error_type = np.where(cond_tp, "TP",
                   np.where(cond_tn, "TN",
                   np.where(cond_fp, "FP",
                   np.where(cond_fn, "FN", "Other"))))
    proj_df["error_type"] = error_type

    proj_csv = os.path.join(out_dir, f"pca_projection_{prefix}.csv")
    proj_df.to_csv(proj_csv, index=False)

    ev_df = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    ev_csv = os.path.join(out_dir, f"pca_explained_variance_{prefix}.csv")
    ev_df.to_csv(ev_csv, index=False)

    bands = np.arange(Xb.shape[1])
    comp_df = pd.DataFrame({
        "band_index": bands,
        "PC1_loading": pca.components_[0, :],
        "PC2_loading": pca.components_[1, :]
    })
    comp_csv = os.path.join(out_dir, f"pca_components_{prefix}.csv")
    comp_df.to_csv(comp_csv, index=False)

    top_rows = []
    for j, pc in enumerate(["PC1", "PC2"]):
        load = comp_df[f"{pc}_loading"].values
        order = np.argsort(np.abs(load))[::-1][:PCA_TOP_LOADINGS]
        for rank, idx_band in enumerate(order, start=1):
            top_rows.append({
                "component": pc,
                "rank": rank,
                "band_index": int(comp_df.loc[idx_band, "band_index"]),
                "loading": float(load[idx_band]),
                "abs_loading": float(abs(load[idx_band]))
            })
    top_df = pd.DataFrame(top_rows)
    top_csv = os.path.join(out_dir, f"pca_top_loadings_{prefix}.csv")
    top_df.to_csv(top_csv, index=False)

    plt.figure(figsize=(7.2, 6))
    palette = {0: "#1f77b4", 1: "#d62728"}  # azul / vermelho
    markers = {"TP": "o", "TN": "o", "FP": "x", "FN": "x"}
    edgecolors = {"TP": "none", "TN": "none", "FP": "k", "FN": "k"}

    for etype in ["TP", "TN", "FP", "FN"]:
        mask = proj_df["error_type"] == etype
        if mask.any():
            plt.scatter(
                proj_df.loc[mask, "PC1"],
                proj_df.loc[mask, "PC2"],
                c=proj_df.loc[mask, "true_label"].map(palette),
                marker=markers[etype],
                edgecolors=edgecolors[etype],
                alpha=0.35,
                s=14,
                label=etype
            )

    plt.xlabel("PC1 (standardized features)")
    plt.ylabel("PC2 (standardized features)")
    plt.title(f"PCA Projection — {prefix} (n={len(proj_df)})")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend(title="Point type", loc="best", frameon=True)
    plt.tight_layout()
    pca_png = os.path.join(out_dir, f"pca_plot_{prefix}.png")
    plt.savefig(pca_png, dpi=150)
    plt.close()

    return {
        "pca_proj_csv": proj_csv,
        "pca_ev_csv": ev_csv,
        "pca_comp_csv": comp_csv,
        "pca_top_csv": top_csv,
        "pca_png": pca_png
    }

# =============================
# NEW: ANOVA & RF plots per-band
# =============================
def compute_anova_and_plot(X_train, y_train, prefix, out_dir, wavelengths_nm=None, regions_df=None):
    """
    Calcula F-scores ANOVA em X_train (n, bands) e plota F-score por banda.
    Se wavelengths_nm e regions_df existirem, usa eixo em nm e sombreia regiões.
    """
    # Garantir shape 2D
    if X_train.ndim == 3 and X_train.shape[-1] == 1:
        Xb = X_train.squeeze(-1)
    else:
        Xb = X_train

    f_values, _ = f_classif(Xb, y_train)
    # Normaliza para [0,1] para facilitar leitura
    if np.max(f_values) > 0:
        f_norm = f_values / np.max(f_values)
    else:
        f_norm = f_values

    # Salvar CSV
    if wavelengths_nm is not None and len(wavelengths_nm) == Xb.shape[1]:
        df = pd.DataFrame({"band_index": np.arange(Xb.shape[1]),
                           "wavelength_nm": wavelengths_nm,
                           "anova_fscore": f_values,
                           "anova_fscore_norm": f_norm})
        x_axis = wavelengths_nm
        x_label = "Wavelength (nm)"
        use_nm = True
    else:
        if wavelengths_nm is not None:
            print("⚠️ wavelengths length does not match number of bands — using band index axis.")
        df = pd.DataFrame({"band_index": np.arange(Xb.shape[1]),
                           "anova_fscore": f_values,
                           "anova_fscore_norm": f_norm})
        x_axis = np.arange(Xb.shape[1])
        x_label = "Band index"
        use_nm = False
    csv_path = os.path.join(out_dir, f"anova_f_scores_{prefix}.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(10, 4.8))
    plt.plot(x_axis, f_norm, label="ANOVA F-score (normalized)")
    plt.xlabel(x_label)
    plt.ylabel("Normalized F-score (a.u.)")
    plt.title(f"Distribution of ANOVA F-scores by spectral band — {prefix}")
    plt.grid(True, linestyle='--', linewidth=0.5)
    # Sombreia regiões se possível
    if use_nm and regions_df is not None and not regions_df.empty:
        for _, row in regions_df.iterrows():
            try:
                start_nm = float(row["start_nm"])
                end_nm = float(row["end_nm"])
                label = str(row["label"])
                plt.axvspan(start_nm, end_nm, color='orange', alpha=0.15)
                # marca label no topo
                mid = 0.5 * (start_nm + end_nm)
                ymax = np.nanmax(f_norm) if f_norm.size else 1.0
                plt.text(mid, ymax, label, ha='center', va='bottom', fontsize=8, rotation=0)
            except Exception:
                pass
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"anova_f_scores_{prefix}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"💾 ANOVA saved for '{prefix}':")
    print(f"   - CSV: {csv_path}")
    print(f"   - PNG: {png_path}")
    return csv_path, png_path


def compute_rf_and_plot(X_train, y_train, prefix, out_dir, top_k=RF_TOP_K, wavelengths_nm=None):
    """
    Treina RF em X_train (padronizado) e plota importâncias normalizadas por banda.
    Destaca Top-K bandas e salva CSVs.
    """
    # Flatten if needed
    if X_train.ndim == 3 and X_train.shape[-1] == 1:
        Xb = X_train.squeeze(-1)
    else:
        Xb = X_train

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xb)

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(Xs, y_train)
    imp = rf.feature_importances_.astype(float)
    imp_norm = imp / np.max(imp) if np.max(imp) > 0 else imp

    # DataFrame de importâncias
    df_cols = {"band_index": np.arange(Xb.shape[1]),
               "rf_importance": imp,
               "rf_importance_norm": imp_norm}
    if wavelengths_nm is not None and len(wavelengths_nm) == Xb.shape[1]:
        df_cols["wavelength_nm"] = wavelengths_nm
        x_axis = wavelengths_nm
        x_label = "Wavelength (nm)"
    else:
        if wavelengths_nm is not None:
            print("⚠️ wavelengths length does not match number of bands — using band index axis.")
        x_axis = np.arange(Xb.shape[1])
        x_label = "Band index"
    df_imp = pd.DataFrame(df_cols)

    # Top-K
    order = np.argsort(imp_norm)[::-1]
    top_idx = order[:min(top_k, len(order))]
    df_top = df_imp.loc[top_idx].sort_values(by="rf_importance_norm", ascending=False)

    # Save CSVs
    imp_csv = os.path.join(out_dir, f"rf_importances_{prefix}.csv")
    top_csv = os.path.join(out_dir, f"rf_top{len(top_idx)}_{prefix}.csv")
    df_imp.to_csv(imp_csv, index=False)
    df_top.to_csv(top_csv, index=False)

    # Plot
    plt.figure(figsize=(10, 4.8))
    plt.plot(x_axis, imp_norm, label="RF normalized importance")
    # highlight top-k
    plt.scatter(x_axis[top_idx], imp_norm[top_idx], s=28, marker='o', edgecolors='k', facecolors='none', label=f"Top-{len(top_idx)} bands")
    plt.xlabel(x_label)
    plt.ylabel("Normalized importance (a.u.)")
    plt.title(f"Spectral bands estimated by Random Forest — {prefix}")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"rf_importance_{prefix}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"💾 RF importance saved for '{prefix}':")
    print(f"   - Importances CSV: {imp_csv}")
    print(f"   - Top-{len(top_idx)} CSV: {top_csv}")
    print(f"   - PNG: {png_path}")
    return imp_csv, top_csv, png_path

# =============================
# Evaluation
# =============================
def evaluate_one(cfg):
    label = cfg["label"]
    prefix = cfg["name"].lower()
    print(f"\n🚀 Processing: {label.upper()}")

    # Output subdir
    subdir = os.path.join(OUT_PATH, prefix)
    os.makedirs(subdir, exist_ok=True)

    # Load data
    X = np.load(os.path.join(DATA_PATH, cfg["x_file"]))
    y = np.load(os.path.join(DATA_PATH, cfg["y_file"]))
    X = ensure_channel_dim(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Load wavelengths & regions (optional)
    wavelengths_nm = load_wavelengths_if_any()
    regions_df = load_regions_if_any()

    # Load model
    model_path = os.path.join(DATA_PATH, cfg["model_file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Predict
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }

    # Confusions
    cm = confusion_matrix(y_test, y_pred)
    cm_paths = plot_confusions_and_save(cm, CLASS_NAMES, prefix, subdir)

    # ROC & PR
    roc_png, roc_csv, roc_auc = plot_save_roc(y_test, y_proba, prefix, subdir)
    pr_png, pr_csv, ap = plot_save_pr(y_test, y_proba, prefix, subdir)
    metrics["Average_Precision"] = ap

    # Save metrics
    metrics_csv = os.path.join(subdir, f"metrics_{prefix}.csv")
    metrics_json = os.path.join(subdir, f"metrics_{prefix}.json")
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Classification report
    report_txt = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    report_file = os.path.join(subdir, f"classification_report_{prefix}.txt")
    with open(report_file, "w") as f:
        f.write(report_txt)

    # History (optional)
    hist = safe_load_history(os.path.join(DATA_PATH, cfg["history_pkl"]))
    hist_fig_path = plot_save_history(hist, prefix, subdir) if hist else None

    # Error spectra
    err_csv, err_png = compute_and_save_error_spectra(X_test, y_test, y_pred, prefix, subdir)

    # PCA
    pca_paths = pca_analysis_save(X_test, y_test, y_pred, prefix, subdir)

    # NEW: ANOVA + RF (usando APENAS o conjunto de TREINO para evitar vazamento)
    anova_csv, anova_png = compute_anova_and_plot(X_train, y_train, prefix, subdir, wavelengths_nm, regions_df)
    rf_imp_csv, rf_top_csv, rf_png = compute_rf_and_plot(X_train, y_train, prefix, subdir, RF_TOP_K, wavelengths_nm)

    return {
        "name": label,
        "prefix": prefix,
        "subdir": subdir,
        "metrics": metrics,
        "metrics_csv": metrics_csv,
        "metrics_json": metrics_json,
        "report_file": report_file,
        "roc_png": roc_png, "roc_csv": roc_csv,
        "pr_png": pr_png, "pr_csv": pr_csv,
        "hist_fig_path": hist_fig_path,
        "error_spectra_csv": err_csv,
        "error_spectra_png": err_png,
        **cm_paths,
        **(pca_paths if pca_paths else {}),
        "anova_csv": anova_csv,
        "anova_png": anova_png,
        "rf_importances_csv": rf_imp_csv,
        "rf_top_csv": rf_top_csv,
        "rf_png": rf_png,
    }


def write_index(results):
    idx_path = os.path.join(OUT_PATH, "artifact_index.md")
    lines = []
    lines.append("# Analysis Index")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    for res in results:
        lines.append(f"## {res['name']} model")
        m = res["metrics"]
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Accuracy          | {m['Accuracy']:.4f} |")
        lines.append(f"| Precision         | {m['Precision']:.4f} |")
        lines.append(f"| Recall            | {m['Recall']:.4f} |")
        lines.append(f"| F1-Score          | {m['F1_Score']:.4f} |")
        lines.append(f"| ROC AUC           | {m['ROC_AUC']:.4f} |")
        lines.append(f"| Average Precision | {m['Average_Precision']:.4f} |")
        lines.append("")
        lines.append(f"- **Metrics (CSV):** `{res['metrics_csv']}`")
        lines.append(f"- **Metrics (JSON):** `{res['metrics_json']}`")
        lines.append(f"- **Classification report:** `{res['report_file']}`")
        lines.append(f"- **Confusion matrix (counts, PNG):** `{res['cm_counts_png']}`")
        lines.append(f"- **Confusion matrix (counts, CSV):** `{res['cm_counts_csv']}`")
        lines.append(f"- **Confusion matrix (row-normalized %, PNG):** `{res['cm_norm_png']}`")
        lines.append(f"- **Confusion matrix (row-normalized %, CSV):** `{res['cm_norm_csv']}`")
        lines.append(f"- **ROC curve:** `{res['roc_png']}`")
        lines.append(f"- **ROC points (CSV):** `{res['roc_csv']}`")
        lines.append(f"- **Precision-Recall curve:** `{res['pr_png']}`")
        lines.append(f"- **Precision-Recall points (CSV):** `{res['pr_csv']}`")
        if res.get("hist_fig_path"):
            lines.append(f"- **History (loss/accuracy):** `{res['hist_fig_path']}`")
        if res.get("error_spectra_csv"):
            lines.append(f"- **Error spectra (CSV):** `{res['error_spectra_csv']}`")
        if res.get("error_spectra_png"):
            lines.append(f"- **Error spectra (PNG):** `{res['error_spectra_png']}`")
        if res.get("pca_proj_csv"):
            lines.append(f"- **PCA projection (CSV):** `{res['pca_proj_csv']}`")
        if res.get("pca_ev_csv"):
            lines.append(f"- **PCA explained variance (CSV):** `{res['pca_ev_csv']}`")
        if res.get("pca_comp_csv"):
            lines.append(f"- **PCA components/loadings (CSV):** `{res['pca_comp_csv']}`")
        if res.get("pca_top_csv"):
            lines.append(f"- **PCA top-|loading| bands (CSV):** `{res['pca_top_csv']}`")
        if res.get("pca_png"):
            lines.append(f"- **PCA plot (PNG):** `{res['pca_png']}`")
        # NEW references
        lines.append(f"- **ANOVA F-scores (CSV):** `{res['anova_csv']}`")
        lines.append(f"- **ANOVA F-scores (PNG):** `{res['anova_png']}`")
        lines.append(f"- **RF importances (CSV):** `{res['rf_importances_csv']}`")
        lines.append(f"- **RF top-{RF_TOP_K} bands (CSV):** `{res['rf_top_csv']}`")
        lines.append(f"- **RF importance plot (PNG):** `{res['rf_png']}`")
        lines.append("")

    with open(idx_path, "w") as f:
        f.write("\n".join(lines))
    return idx_path


def main():
    print(f"📂 Output directory: {OUT_PATH}")
    results = []
    for cfg in CONFIGS:
        try:
            res = evaluate_one(cfg)
            results.append(res)
        except Exception as e:
            print(f"❌ Error in {cfg['label']}: {e}")

    idx = write_index(results)
    print(f"🗂️  Index saved at: {idx}")


if __name__ == "__main__":
    main()

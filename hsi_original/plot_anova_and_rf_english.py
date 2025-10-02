#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-axes ANOVA F-score plot (English labels, grid) with background shaded regions.
Saves to ./figures/anova_f_scores_en.png

Exemplos:
# Com comprimentos de onda (nm)
python3 plot_anova_f_scores_en_simple.py \
  --data-path ./hsi_cnn \
  --anova-file anova_f_scores.csv \
  --wavelengths-file wavelengths.csv

# Sem wavelengths (usa índice de banda; mapeia nm com nm-min/nm-max)
python3 plot_anova_f_scores_en_simple.py \
  --data-path ./hsi_cnn \
  --anova-file anova_f_scores.csv \
  --nm-min 1000 --nm-max 2500
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def load_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path)).ravel().astype(float)
    if path.suffix.lower() == ".csv":
        # tenta sem e com header; depois última coluna
        for kw in ({}, {"skiprows": 1}):
            try:
                arr = np.loadtxt(path, delimiter=",", dtype=float, **kw)
                return np.asarray(arr).ravel().astype(float)
            except Exception:
                pass
        data = np.genfromtxt(path, delimiter=",")
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            return data.ravel()
        return data[:, -1].ravel()
    raise ValueError(f"Unsupported file type: {path.suffix}")

def maybe_load(path: Path) -> np.ndarray | None:
    return load_vector(path) if path.exists() else None

def parse_regions(spec: str):
    out = []
    if not spec:
        return out
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            try:
                out.append((float(a), float(b)))
            except ValueError:
                pass
    return out

def is_nm_axis(arr: np.ndarray) -> bool:
    if np.nanmax(arr) < 500:
        return False
    diffs = np.diff(arr[np.isfinite(arr)])
    return np.all(diffs >= 0)

def nm_to_index(nm_val: float, nm_min: float, nm_max: float, n_bands: int) -> float:
    if nm_max <= nm_min:
        return 0.0
    t = (nm_val - nm_min) / (nm_max - nm_min)
    t = np.clip(t, 0.0, 1.0)
    return t * (n_bands - 1)

def normalize01(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return np.zeros_like(y)
    lo, hi = float(np.min(y[finite])), float(np.max(y[finite]))
    if hi <= lo:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)

def main():
    ap = argparse.ArgumentParser(description="Single-axes ANOVA F-score plot with shaded nm regions.")
    ap.add_argument("--data-path", type=Path, default=Path("./hsi_cnn"))
    ap.add_argument("--anova-file", type=str, default="anova_f_scores.csv")
    ap.add_argument("--wavelengths-file", type=str, default="wavelengths.csv")
    ap.add_argument("--nm-min", type=float, default=1000.0)
    ap.add_argument("--nm-max", type=float, default=2500.0)
    ap.add_argument("--regions", type=str, default="1200-1800,2100-2300",
                    help='Comma-separated nm ranges, e.g. "1200-1800,2100-2300"')
    ap.add_argument("--out-name", type=str, default="anova_f_scores_en.png")
    args = ap.parse_args()

    out_dir = Path("./figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    f = load_vector(args.data_path / args.anova_file)
    f = normalize01(f)
    n = len(f)

    wl = maybe_load(args.data_path / args.wavelengths_file)
    if wl is not None and len(wl) == n and is_nm_axis(wl):
        x = wl.astype(float)
        x_label = "Wavelength (nm)"
        x_mode_nm = True
    else:
        x = np.arange(n, dtype=float)
        x_label = "Band index"
        x_mode_nm = False

    regions_nm = parse_regions(args.regions)
    if not regions_nm:
        regions_nm = [(1200.0, 1800.0), (2100.0, 2300.0)]

    # ---- Plot (single axis) ----
    fig, ax = plt.subplots(figsize=(12, 4.6))

    # 1) sombrear faixas no *fundo* (zorder baixo) ANTES da curva
    colors = [(1.0, 0.6, 0.6, 0.22), (0.6, 0.8, 0.6, 0.22)]
    labels_bg = []
    for i, (lo_nm, hi_nm) in enumerate(regions_nm):
        if x_mode_nm:
            lo, hi = lo_nm, hi_nm
        else:
            lo = nm_to_index(lo_nm, args.nm_min, args.nm_max, n)
            hi = nm_to_index(hi_nm, args.nm_min, args.nm_max, n)
        if hi < lo:
            lo, hi = hi, lo
        ax.axvspan(lo, hi, facecolor=colors[i % len(colors)], edgecolor="none",
                   zorder=0, linewidth=0)
        labels_bg.append(f"{int(lo_nm)}–{int(hi_nm)} nm")

    # 2) curva por cima, zorder alto + contorno leve
    (line,) = ax.plot(x, f, linewidth=2.6, color="#1f77b4",
                      label="F-score per spectral band", zorder=5)
    line.set_path_effects([pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()])

    # 3) ajustes visuais
    ax.set_title("Distribution of ANOVA F-scores by Spectral Band")
    ax.set_xlabel(x_label)
    ax.set_ylabel("F-score (ANOVA)")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(0.0, max(1.02, float(np.nanmax(f) * 1.05)))

    # 4) legenda (linha + texto das regiões)
    # (não adicionamos os spans na legenda para mantê-la limpa)
    ax.legend(loc="best", title="Curves")

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")
    print(f"[INFO] X-axis: {'nm' if x_mode_nm else 'band index'} | Regions: {', '.join(labels_bg)}")

if __name__ == "__main__":
    main()

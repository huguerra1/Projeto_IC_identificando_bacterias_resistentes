#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:12:25 2025

@author: clarimar
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
import spectral

# ======================
# BASE DE DIRETÓRIOS (independente do CWD)
# ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "hsi_original"
OUTPUT_DIR = BASE_DIR / "hsi_cnn"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"[DEBUG] DATA_ROOT={DATA_ROOT} exists={DATA_ROOT.exists()}")

# ======================
# UTILITÁRIOS DE ARQUIVOS
# ======================
def list_sample_dirs(root: Path):
    """Lista apenas diretórios que aparentam ser amostras (devem conter 'capture')."""
    dirs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "capture").exists():
            dirs.append(p)
        else:
            # Skip silencioso para pastas como hsi_cnn/ e arquivos soltos
            pass
    print(f"[DEBUG] {len(dirs)} amostras encontradas.")
    return dirs

def get_meta_dir(sample_dir: Path) -> Path | None:
    """Aceita 'metadata' e 'metada' por compatibilidade."""
    for name in ("metadata", "metada"):
        cand = sample_dir / "capture" / name
        if cand.exists():
            return cand
    return None

def find_capture_files(capture_dir: Path):
    """
    Varre capture/ e encontra:
      - RAW principal (não começa com DARKREF/WHITEREF)
      - DARKREF*.raw
      - WHITEREF*.raw
    Assume que .hdr tem o mesmo nome trocando a extensão.
    Retorna tupla de paths: (raw_hdr, raw_bin, dark_hdr, dark_bin, white_hdr, white_bin)
    """
    raws = list(capture_dir.rglob("*.raw")) + list(capture_dir.rglob("*.img"))  # alguns datasets usam .img
    if not raws:
        raise FileNotFoundError(f"Nenhum arquivo .raw/.img encontrado em {capture_dir}")

    # arquivos de referência
    dark = next((p for p in raws if p.name.upper().startswith("DARKREF")), None)
    white = next((p for p in raws if p.name.upper().startswith("WHITEREF")), None)

    # arquivo principal = primeiro .raw/.img que não é DARKREF/WHITEREF
    main = next((p for p in raws if not p.name.upper().startswith(("DARKREF", "WHITEREF"))), None)

    if main is None or dark is None or white is None:
        raise FileNotFoundError(
            f"Arquivos esperados não encontrados em {capture_dir} "
            f"(main={bool(main)}, dark={bool(dark)}, white={bool(white)})"
        )

    def hdr_for(bin_path: Path):
        # tenta mesmo nome trocando extensão
        cand = bin_path.with_suffix(".hdr")
        if cand.exists():
            return cand
        # fallback: qualquer .hdr no mesmo diretório que comece com o stem
        for h in bin_path.parent.glob("*.hdr"):
            if h.stem == bin_path.stem:
                return h
        raise FileNotFoundError(f"HDR correspondente não encontrado para {bin_path}")

    main_hdr = hdr_for(main)
    dark_hdr = hdr_for(dark)
    white_hdr = hdr_for(white)

    return (main_hdr, main, dark_hdr, dark, white_hdr, white)

def load_envi(hdr_path: Path, bin_path: Path):
    """Carrega ENVI usando spectral, garantindo caminho absoluto."""
    img = spectral.envi.open(str(hdr_path), str(bin_path))
    return img.load()  # carrega em array numpy

def calibrate_image_from_capture(capture_dir: Path):
    """
    Calibração: (img - dark) / (white - dark), com proteção anti-divisão por zero.
    Clipa para [0,1].
    """
    main_hdr, main_bin, dark_hdr, dark_bin, white_hdr, white_bin = find_capture_files(capture_dir)

    img = load_envi(main_hdr, main_bin).astype(np.float32)
    dark = load_envi(dark_hdr, dark_bin).astype(np.float32)
    white = load_envi(white_hdr, white_bin).astype(np.float32)

    denom = white - dark
    # proteção contra divisão por zero / negativo
    eps = 1e-6
    denom = np.where(np.abs(denom) < eps, eps, denom)
    calibrated = (img - dark) / denom
    calibrated = np.clip(calibrated, 0.0, 1.0)

    return img, calibrated

# ======================
# LEITURA E PROCESSAMENTO
# ======================
sample_dirs = list_sample_dirs(DATA_ROOT)

X_list, y_list = [], []
images_raw, images_calib = [], []
labels_images = []

for i, sdir in enumerate(tqdm(sample_dirs, desc="Processando amostras")):
    capture_dir = sdir / "capture"
    # meta_dir não é obrigatório, mas guardamos para debug
    _meta_dir = get_meta_dir(sdir)

    try:
        img_raw, img_calib = calibrate_image_from_capture(capture_dir)
    except Exception as e:
        print(f"Erro na amostra {sdir.name}: {e}")
        continue

    pixels = img_calib.reshape(-1, img_calib.shape[-1])
    label = 1 if sdir.name.startswith('ATCC') else 0
    labels = np.full((pixels.shape[0],), label, dtype=np.int8)

    X_list.append(pixels)
    y_list.append(labels)

    # guarda duas para plot
    if len(images_raw) < 2:
        images_raw.append(img_raw)
        images_calib.append(img_calib)
        labels_images.append(label)

if not X_list:
    raise RuntimeError(
        "Nenhuma amostra processada. Verifique se existem arquivos .raw/.hdr em 'capture/' "
        "e se os arquivos DARKREF/WHITEREF estão presentes."
    )

X = np.vstack(X_list)
y = np.hstack(y_list)

np.save(OUTPUT_DIR / 'X.npy', X)
np.save(OUTPUT_DIR / 'y.npy', y)

# ======================
# PLOTS DE IMAGENS
# ======================
def plot_image(image, title, filename):
    plt.figure()
    plt.imshow(np.mean(image, axis=2), cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

for idx in range(len(images_raw)):
    plot_image(images_raw[idx], f'Imagem {idx+1} Original', f'img{idx+1}_original.png')
    plot_image(images_calib[idx], f'Imagem {idx+1} Calibrada', f'img{idx+1}_calibrada.png')

# ======================
# PLOTS DE ESPECTROS
# ======================
plt.figure()
for idx in range(len(images_raw)):
    mean_raw = np.mean(images_raw[idx].reshape(-1, images_raw[idx].shape[-1]), axis=0)
    mean_calib = np.mean(images_calib[idx].reshape(-1, images_calib[idx].shape[-1]), axis=0)
    plt.plot(mean_raw, '--', label=f'Imagem {idx+1} Original')
    plt.plot(mean_calib, '-', label=f'Imagem {idx+1} Calibrada')
plt.title('Espectros Médios')
plt.xlabel('Bandas')
plt.ylabel('Intensidade')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'espectros_medios.png', dpi=150)
plt.close()

# ======================
# SELEÇÃO DE BANDAS
# ======================
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
importances = rf.feature_importances_
idx_rf = np.argsort(importances)[-30:]

f_values, _ = f_classif(X, y)
idx_anova = np.argsort(f_values)[-30:]

idx_selected = np.union1d(idx_rf, idx_anova)
X_selected = X[:, idx_selected]

np.save(OUTPUT_DIR / 'X_selected.npy', X_selected)
np.save(OUTPUT_DIR / 'idx_selected.npy', idx_selected)

# ======================
# PLOT IMAGENS APÓS SELEÇÃO
# ======================
def plot_selected(image, title, filename, idx_selected):
    reduced = image[:, :, idx_selected].mean(axis=2)
    plt.figure()
    plt.imshow(reduced, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

for idx in range(len(images_calib)):
    plot_selected(images_calib[idx], f'Imagem {idx+1} Selecionada', f'img{idx+1}_selecionada.png', idx_selected)

print("\n✅ Dataset gerado com sucesso. Arquivos salvos em ./hsi_cnn/")

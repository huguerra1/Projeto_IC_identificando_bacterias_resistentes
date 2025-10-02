#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# spectral ENVI
import spectral as sp
sp.settings.envi_support_nonlowercase_params = True  # tolerate non-lowercase header keys


# =============================
# PATHS (edit if needed)
# =============================
DATA_PATH = Path('./hsi_original/')
SAVE_PATH = Path('./hsi_cnn/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


# =============================
# HELPERS
# =============================
def find_hdr_for(bin_path: Path) -> Path:
    """Return the matching .hdr for a given .raw/.img file."""
    cand = bin_path.with_suffix('.hdr')
    if cand.exists():
        return cand
    # fallback: search by stem
    for h in bin_path.parent.glob('*.hdr'):
        if h.stem == bin_path.stem:
            return h
    raise FileNotFoundError(f'HDR not found for {bin_path}')

def load_envi_cube(hdr: Path, binf: Path) -> np.ndarray:
    """Open ENVI pair and return float32 array (L, S, B)."""
    img = sp.envi.open(str(hdr), str(binf))
    arr = img.load()
    if arr is None:
        # materialize defensively
        B = img.nbands
        first = np.array(img.read_band(0))
        L, S = first.shape
        cube = np.empty((L, S, B), dtype=np.float32)
        cube[:, :, 0] = first
        for b in range(1, B):
            cube[:, :, b] = img.read_band(b)
        return cube.astype(np.float32)
    return np.asarray(arr, dtype=np.float32)

def list_sample_folders(root: Path) -> list[Path]:
    """List subfolders that contain a 'capture' directory."""
    if not root.exists():
        raise FileNotFoundError(f'Data path does not exist: {root}')
    samples = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / 'capture').exists():
            samples.append(p)
    return samples

def find_capture_files(capture_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """
    Find ENVI binaries for main, DARKREF, WHITEREF inside capture_dir.
    Accepts .raw or .img. Returns (main_hdr, main_bin, dark_hdr, dark_bin, white_hdr, white_bin).
    """
    if not capture_dir.exists():
        raise FileNotFoundError(f"'capture' not found: {capture_dir}")

    bins = list(capture_dir.rglob('*.raw')) + list(capture_dir.rglob('*.img'))
    if not bins:
        raise FileNotFoundError(f'No .raw/.img files found in {capture_dir}')

    def is_dark(p: Path) -> bool: return p.name.upper().startswith('DARKREF')
    def is_white(p: Path) -> bool: return p.name.upper().startswith('WHITEREF')

    dark  = next((p for p in bins if is_dark(p)), None)
    white = next((p for p in bins if is_white(p)), None)
    main  = next((p for p in bins if not (is_dark(p) or is_white(p))), None)

    if main is None or dark is None or white is None:
        raise FileNotFoundError(
            f'Expected main/DARKREF/WHITEREF not all found in {capture_dir} '
            f'(main={bool(main)}, dark={bool(dark)}, white={bool(white)})'
        )

    return (
        find_hdr_for(main),  main,
        find_hdr_for(dark),  dark,
        find_hdr_for(white), white
    )

def calibrate_with_band_means(raw: np.ndarray, dark: np.ndarray, white: np.ndarray) -> np.ndarray:
    """
    Robust calibration allowing different spatial shapes:
      dark_vec[b]  = mean(dark[:, :, b])
      white_vec[b] = mean(white[:, :, b])
      corrected    = clip( (raw - dark_vec) / max(white_vec - dark_vec, eps), 0, 1 )
    """
    reducer = np.mean
    dark_vec  = reducer(dark,  axis=(0, 1))  # (B,)
    white_vec = reducer(white, axis=(0, 1))  # (B,)

    # Broadcast to raw shape (L, S, B)
    dark_b  = np.broadcast_to(dark_vec,  raw.shape)
    white_b = np.broadcast_to(white_vec, raw.shape)

    denom = white_b - dark_b
    eps = 1e-6
    denom = np.where(np.abs(denom) < eps, eps, denom)

    corrected = (raw - dark_b) / denom
    corrected = np.clip(corrected, 0.0, 1.0)
    corrected = np.nan_to_num(corrected, nan=0.0, posinf=1.0, neginf=0.0)
    return corrected


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    sample_folders = list_sample_folders(DATA_PATH)

    print(f'\nScanning samples in: {DATA_PATH}')
    print(f'Found {len(sample_folders)} sample(s): {[p.name for p in sample_folders]}')

    X_list, y_list = [], []

    for sample_dir in tqdm(sample_folders, desc='Processing samples'):
        try:
            capture = sample_dir / 'capture'

            main_hdr, main_bin, dark_hdr, dark_bin, white_hdr, white_bin = find_capture_files(capture)

            raw   = load_envi_cube(main_hdr,  main_bin)   # (L, S, B)
            dark  = load_envi_cube(dark_hdr,  dark_bin)   # (L', S', B)
            white = load_envi_cube(white_hdr, white_bin)  # (L'', S'', B)

            corrected = calibrate_with_band_means(raw, dark, white)

            # reshape to (pixels, bands)
            pixels = corrected.reshape(-1, corrected.shape[-1]).astype(np.float32)
            X_list.append(pixels)

            # label: 1 if folder starts with 'ATCC', else 0
            label = 1 if sample_dir.name.upper().startswith('ATCC') else 0
            y_list.append(np.full((pixels.shape[0],), label, dtype=np.int8))

        except Exception as e:
            print(f'❌ Error in sample {sample_dir.name}: {e}')

    if not X_list:
        raise RuntimeError('No samples processed. Check file structure and ENVI files in each capture.')

    # =============================
    # STACK FULL DATASET
    # =============================
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    print(f'\nSaving full dataset:')
    print(f' - X shape: {X.shape}')
    print(f' - y shape: {y.shape}')

    np.save(SAVE_PATH / 'X.npy', X)
    np.save(SAVE_PATH / 'y.npy', y)

    # =============================
    # TRAIN/TEST SPLIT
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f'\nTrain/Test split:')
    print(f' - X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f' - X_test : {X_test.shape},  y_test : {y_test.shape}')

    np.save(SAVE_PATH / 'X_train.npy', X_train)
    np.save(SAVE_PATH / 'y_train.npy', y_train)
    np.save(SAVE_PATH / 'X_test.npy', X_test)
    np.save(SAVE_PATH / 'y_test.npy', y_test)

    print('\n✅ Dataset built and saved successfully!')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:07:10 2025

@author: clarimar
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# === 🔧 Função para ler o .hdr ===
def parse_envi_hdr(hdr_path):
    """Lê e interpreta um arquivo .hdr no formato ENVI."""
    metadata = {}
    with open(hdr_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip().lower()

            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1].strip()

            try:
                if '.' in value:
                    metadata[key] = float(value)
                else:
                    metadata[key] = int(value)
            except:
                metadata[key] = value

    return metadata


# === 📄 Arquivos
raw_file = '351_1_240506-160920.raw'
hdr_file = '351_1_240506-160920.hdr'

# === 🔍 Ler metadados
meta = parse_envi_hdr(hdr_file)

samples = int(meta.get('samples'))
lines = int(meta.get('lines'))
bands = int(meta.get('bands'))
interleave = meta.get('interleave').lower()

# === ✔️ Corrigir manualmente o data_type
data_type = 2  # uint16 (corrigindo do header incorreto)
byte_order = int(meta.get('byte order', 0))

print('Metadados extraídos:')
print(f'Samples (colunas): {samples}')
print(f'Lines (linhas): {lines}')
print(f'Bands: {bands}')
print(f'Interleave: {interleave}')
print(f'Data type (forçado): {data_type}')
print(f'Byte order: {byte_order}')

# === 📦 Mapeamento correto de data type
data_type_map = {
    1: np.uint8,
    2: np.uint16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.float32
}

if data_type not in data_type_map:
    raise ValueError(f'Data type {data_type} não suportado.')

np_dtype = np.dtype(data_type_map[data_type])

# === 🔗 Ajustar byte order
if byte_order == 0:
    np_dtype = np_dtype.newbyteorder('<')  # little endian
else:
    np_dtype = np_dtype.newbyteorder('>')  # big endian

# === 🚀 Ler dados
data = np.fromfile(raw_file, dtype=np_dtype)

expected_size = samples * lines * bands
assert data.size == expected_size, f"Tamanho incompatível. Esperado {expected_size}, obtido {data.size}"

# === 🔄 Rearranjar conforme interleave
if interleave == 'bil':
    data_reshaped = data.reshape((lines, bands, samples))
    hsi_image = np.transpose(data_reshaped, (0, 2, 1))  # (lines, samples, bands)
elif interleave == 'bsq':
    data_reshaped = data.reshape((bands, lines, samples))
    hsi_image = np.transpose(data_reshaped, (1, 2, 0))
elif interleave == 'bip':
    hsi_image = data.reshape((lines, samples, bands))
else:
    raise ValueError(f'Interleave {interleave} não reconhecido.')

print(f'Imagem carregada com sucesso. Shape: {hsi_image.shape}')

# =========================================================
# === 🖼️ Plot de uma banda intermediária com mapa de cores ===
# =========================================================
banda_idx = bands // 2  # banda intermediária

plt.figure(figsize=(6,6))
plt.imshow(hsi_image[:, :, banda_idx], cmap='jet')  # 🌈 Usando mapa de cores
plt.title(f'Banda {banda_idx} - Colormap: Jet')
plt.colorbar(label='Intensidade')
plt.axis('off')
plt.show()

# =========================================================
# === 🎨 Plot de uma composição RGB simulando visão humana ===
# =========================================================
if bands >= 3:
    # Escolher bandas para RGB
    # 💡 Ajuste conforme seu espectro, este é um exemplo comum:
    banda_r = min(bands - 1, bands // 2 + 20)  # Banda simulando vermelho (~650 nm)
    banda_g = bands // 2                       # Banda simulando verde (~550 nm)
    banda_b = max(0, bands // 2 - 20)           # Banda simulando azul (~450 nm)

    rgb = np.stack([
        hsi_image[:, :, banda_r],
        hsi_image[:, :, banda_g],
        hsi_image[:, :, banda_b]
    ], axis=2)

    # Normalizar para 0-1
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-9)

    plt.figure(figsize=(6,6))
    plt.imshow(rgb_norm)
    plt.title(f'Imagem RGB simulada - Bandas (R={banda_r}, G={banda_g}, B={banda_b})')
    plt.axis('off')
    plt.show()

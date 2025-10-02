#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 10:59:18 2025

@author: clarimar
"""

import numpy as np
import matplotlib.pyplot as plt

# Caminho do arquivo .raw
file_path = "351_1_240506-160920.raw"

# Metadados do cabeçalho
samples = 320  # largura
lines = 336    # altura
bands = 256
interleave = 'bil'
dtype = np.uint16

# Carrega os dados como vetor
data = np.fromfile(file_path, dtype=dtype)

# Verifica o tamanho
expected_size = samples * lines * bands
assert data.size == expected_size, f"Erro no tamanho: {data.size} != {expected_size}"

# Converte para imagem hiperespectral com formato BIL (Band Interleaved by Line)
# A ordem correta no reshape para BIL é: (lines, bands, samples)
data_reshaped = data.reshape((lines, bands, samples))

# Rearranja para formato mais comum (lines, samples, bands)
hsi_image = np.transpose(data_reshaped, (0, 2, 1))  # agora está (altura, largura, bandas)

# Plotar uma banda intermediária
banda_idx = 128
plt.imshow(hsi_image[:, :, banda_idx], cmap='gray')
plt.title(f'Banda {banda_idx}')
plt.axis('off')
plt.show()

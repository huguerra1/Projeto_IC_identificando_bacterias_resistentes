# =============================
# 🚀 Deploy CNN 1D — Full e Selected
# =============================

import os
import numpy as np
import tensorflow as tf

# =============================
# ⚙️ Configuração
# =============================
# Escolha 'full' ou 'selected'
MODE = 'full'  # ou 'selected'

DATA_PATH = './hsi_cnn/'

# =============================
# 📦 Definir arquivos conforme o modo
# =============================
if MODE == 'full':
    model_file = os.path.join(DATA_PATH, 'model_cnn1d_full.keras')
    X_file = os.path.join(DATA_PATH, 'X.npy')
    output_prefix = 'full'
elif MODE == 'selected':
    model_file = os.path.join(DATA_PATH, 'model_cnn1d_selected.keras')
    X_file = os.path.join(DATA_PATH, 'X_selected.npy')
    output_prefix = 'selected'
else:
    raise ValueError("❌ Modo inválido. Use 'full' ou 'selected'.")

# =============================
# 🔍 Verificar existência dos arquivos
# =============================
if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ Arquivo de modelo não encontrado: {model_file}")

if not os.path.exists(X_file):
    raise FileNotFoundError(f"❌ Arquivo de dados não encontrado: {X_file}")

# =============================
# 🚀 Carregar modelo
# =============================
print(f"📥 Carregando modelo: {model_file}")
model = tf.keras.models.load_model(model_file)
print("✅ Modelo carregado com sucesso.")

# =============================
# 🚚 Carregar dados
# =============================
X = np.load(X_file)
if X.ndim == 2:
    X = X[..., np.newaxis]

print(f"✅ Dados carregados: X shape {X.shape}")

# =============================
# 🔮 Fazer predições
# =============================
print("🚀 Fazendo predições...")
y_pred_proba = model.predict(X).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

# =============================
# 📝 Mostrar resultados
# =============================
print("\n📜 Resultados das predições:")
for i, (pred, prob) in enumerate(zip(y_pred, y_pred_proba)):
    classe = 'ATCC' if pred == 1 else 'Staphylococcus spp.'
    print(f"Amostra {i}: {classe} (Probabilidade ATCC: {prob:.4f})")

# =============================
# 💾 Salvar resultados
# =============================
np.save(f'predictions_{output_prefix}.npy', y_pred)
np.save(f'predictions_proba_{output_prefix}.npy', y_pred_proba)

print(f"\n✅ Predições salvas como predictions_{output_prefix}.npy e predictions_proba_{output_prefix}.npy")

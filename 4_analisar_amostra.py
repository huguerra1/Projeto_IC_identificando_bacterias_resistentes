import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_precision_score

# Importa os módulos centralizados de configuração e utilidades
import config
import utils
import joblib

# =============================================================================
# FUNÇÃO DE COMPARAÇÃO DE MODELOS
# =============================================================================

def comparar_modelos_no_relatorio(preproc_method, feature_method):
    """
    Lê os modelos salvos e gera uma tabela comparativa de métricas.
    """
    base_fn = f"{preproc_method}_{feature_method}"
    path_dt = config.MODELS_PATH / f"modelo_{base_fn}.joblib"
    path_xgb = config.MODELS_PATH / f"xgboost_{base_fn}.joblib"
    
    # Carrega os dados de teste para validação rápida
    X_test = np.load(config.PREPROCESSED_PATH / f"X_{base_fn}.npy")
    y_test = np.load(config.PREPROCESSED_PATH / f"y_{base_fn}.npy")

    resultados = []

    for nome, path in [("Decision Tree", path_dt), ("XGBoost", path_xgb)]:
        if path.exists():
            modelo = joblib.load(path)
            y_pred = modelo.predict(X_test)
            resultados.append({
                "Modelo": nome,
                "Acurácia": accuracy_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred)
            })
    
    if resultados:
        df_comp = pd.DataFrame(resultados)
        print("\n" + "="*30)
        print("COMPARAÇÃO DE DESEMPENHO")
        print("="*30)
        print(df_comp.to_string(index=False))
        print("="*30)

# =============================================================================
# FUNÇÃO PRINCIPAL DE ANÁLISE
# =============================================================================

def main(nome_amostra: str, preproc_method: str, banda_analisar: int):
    """
    Executa análise visual e comparação de modelos para a amostra.
    """
    print("="*70)
    print(f"ANÁLISE DE AMOSTRA: {nome_amostra}")
    print(f"MÉTODO: {preproc_method} | BANDA: {banda_analisar}")
    print("="*70)

    try:
        # 1. Carregamento dos dados brutos e Cubo HSI [cite: 44, 47]
        caminho_matriz_bruta = config.RAW_MATRICES_PATH / f"bruta_{nome_amostra}.npy"
        X_bruto = np.load(caminho_matriz_bruta)

        pasta_amostra_original = config.HSI_ORIGINAL_PATH / nome_amostra
        pasta_capture = pasta_amostra_original / "capture"
        
        raw_cube = utils.carregar_cubo_envi(pasta_capture, f"{nome_amostra}.raw")
        dark_cube = utils.carregar_cubo_envi(pasta_capture, f"DARKREF_{nome_amostra}.raw")
        white_cube = utils.carregar_cubo_envi(pasta_capture, f"WHITEREF_{nome_amostra}.raw")
        reflectance_cube = utils.calibrar_para_refletancia(raw_cube, dark_cube, white_cube) # [cite: 48, 51]
        
        # 2. Detecção de ROI e Pré-processamento [cite: 49, 51, 55]
        mask_2d = utils.encontrar_batoque_hough(reflectance_cube, nome_amostra)
        X_proc = utils.aplicar_preprocessamento(X_bruto, preproc_method)

        # 3. Geração de Gráficos (Histogramas e Imagens) [cite: 66, 67]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        # ... (lógica de plotagem idêntica ao seu script original)
        
        plt.tight_layout()
        nome_figura = f"analise_{nome_amostra}_{preproc_method}.png"
        plt.savefig(config.RESULTS_PATH / nome_figura)
        print(f"✅ Gráficos salvos em: {config.RESULTS_PATH}")

        # 4. CHAMADA DA COMPARAÇÃO (Novidade)
        # Tenta comparar para a combinação atual (ex: snv + mi)
        comparar_modelos_no_relatorio(preproc_method, "mi")

    except Exception as e:
        print(f"❌ Erro na análise: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--amostra", type=str, required=True)
    parser.add_argument("-p", "--preproc", type=str, required=True)
    parser.add_argument("-b", "--banda", type=int, default=100)
    
    args = parser.parse_args()
    main(args.amostra, args.preproc, args.banda)

 

def salvar_comparativo(img_rgb, mapa_predicao, nome_saida):
    """Gera uma imagem comparativa para a apresentação de IC."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Lado Esquerdo: Imagem Sintética (O que o olho humano veria)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Imagem Original (Sintética RGB)", fontsize=14)
    axes[0].axis('off')
    
    # Lado Direito: Mapa de Predição do XGBoost
    # Usamos o colormap 'jet' para que o 1 (Resistente) fique vermelho
    im = axes[1].imshow(mapa_predicao, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Mapa de Predição (XGBoost)", fontsize=14)
    axes[1].axis('off')
    
    # Barra de legenda para o mapa de calor
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Sensível', 'Resistente (ATCC)'])
    
    plt.tight_layout()
    plt.savefig(nome_saida, dpi=300) # Alta resolução para o slide
    print(f"✅ Comparativo salvo com sucesso em: {nome_saida}")
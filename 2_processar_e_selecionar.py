import argparse
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

import config
import utils

# Caminho de salvamento manual garantido
FINAL_DATA_PATH = Path("dados_gerados/02_dados_finais")

def main():
    parser = argparse.ArgumentParser(description="Processamento e Seleção de Features HSI")
    parser.add_argument("--preproc", type=str, required=True, help="Método de pré-processamento")
    parser.add_argument("--feature", type=str, required=True, help="Método de seleção/extração")
    args = parser.parse_args()

    print("\n" + "="*70)
    print(f"ETAPA 2: PROCESSAMENTO E SELEÇÃO")
    print(f"Cenário atual -> Pré-proc: {args.preproc.upper()} | Extração: {args.feature.upper()}")
    print("="*70)

    utils.criar_diretorios_necessarios()
    FINAL_DATA_PATH.mkdir(parents=True, exist_ok=True)

    X_list = []
    y_list = []

    # 1. Carregamento e Extração de Pixels
    print("\n-> Carregando amostras e extraindo ROI...")
    pastas_amostras = [p for p in config.HSI_ORIGINAL_PATH.iterdir() if p.is_dir()]
    
    for pasta in tqdm(pastas_amostras, desc="Processando placas"):
        nome_amostra = pasta.name
        rotulo = utils.inferir_rotulo(nome_amostra)
        pasta_capture = pasta / "capture"
        
        # Flexibilidade para o nome do arquivo .raw
        raw_filename = f"{nome_amostra}.raw"
        if not (pasta_capture / raw_filename).exists():
            raw_filename = "capture.raw" # Tenta nome alternativo
            
        try:
            raw = utils.carregar_cubo_envi(pasta_capture, raw_filename)
            dark = utils.carregar_cubo_envi(pasta_capture, f"DARKREF_{raw_filename}")
            white = utils.carregar_cubo_envi(pasta_capture, f"WHITEREF_{raw_filename}")
            
            # Calibração e Máscara
            reflectance = utils.calibrar_para_refletancia(raw, dark, white)
            mask = utils.encontrar_batoque_hough(reflectance, nome_amostra)
            
            # Extrai apenas os pixels dentro do círculo (ROI)
            pixels_roi = reflectance[mask == 1]
            
            X_list.append(pixels_roi)
            y_list.extend([rotulo] * pixels_roi.shape[0])
            
        except Exception as e:
            print(f"\n⚠️ Ignorando amostra {nome_amostra} devido a erro: {e}")

    # Empilha todos os pixels de todas as amostras em uma super matriz
    X_all = np.vstack(X_list)
    y_all = np.array(y_list)
    
    print(f"\n-> Total de pixels extraídos: {X_all.shape[0]}")
    print(f"-> Total de bandas originais: {X_all.shape[1]}")

    # 2. Aplicação do Pré-processamento
    X_preproc = utils.aplicar_preprocessamento(X_all, args.preproc)

    # 3. Aplicação da Seleção/Extração de Features
    X_final, indices = utils.selecionar_features(X_preproc, y_all, args.feature)
    
    print(f"-> Dimensão final dos dados de treino: {X_final.shape}")

    # 4. Salvamento dos Dados
    # O nome do arquivo carrega os métodos usados para não misturar os testes
    nome_base = f"{args.preproc}_{args.feature}"
    
    np.save(FINAL_DATA_PATH / f"X_{nome_base}.npy", X_final)
    np.save(FINAL_DATA_PATH / f"y_{nome_base}.npy", y_all)
    np.save(FINAL_DATA_PATH / f"indices_{nome_base}.npy", indices)

    print(f"\n✅ Dados processados e salvos com sucesso em {FINAL_DATA_PATH}")
    print("="*70)

if __name__ == "__main__":
    main()
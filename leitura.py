import numpy as np
import spectral as sp
from pathlib import Path

# =============================================================================
# CONFIGURAÇÃO (🚨 EDITAR ESTA SEÇÃO)
# =============================================================================

# 1. Defina o caminho para a pasta que contém a sua amostra
DATA_PATH = Path('C:/Users/hugo/OneDrive/identificacao_bacteria/amostras_teste')
# 2. Defina onde o arquivo de saída será salvo
SAVE_PATH = Path('./matrizes_salvas')
SAVE_PATH.mkdir(exist_ok=True) # Cria a pasta se ela não existir

# 3. Informações da amostra que você quer processar
amostra_info = {
    "nome": "351_1_240506-160920/capture",
    "arquivos": {
        "raw": "351_1_240506-160920.raw",
        "dark": "DARKREF_351_1_240506-160920.raw",
        "white": "WHITEREF_351_1_240506-160920.raw"
    },
    "roi_coords": {
        "centro": (100, 150),
        "raio": 45
    }
}

# =============================================================================
# FUNÇÕES AUXILIARES (NÃO PRECISA EDITAR)
# =============================================================================

def carregar_cubo_envi(pasta_amostra, nome_arquivo):
    caminho_raw = pasta_amostra / nome_arquivo
    caminho_hdr = caminho_raw.with_suffix('.hdr')
    if not caminho_hdr.exists():
        raise FileNotFoundError(f"Arquivo HDR não encontrado para {caminho_raw}")
    img = sp.envi.open(str(caminho_hdr), str(caminho_raw))
    return img.load()

def calibrar_para_refletancia(raw_cube, dark_cube, white_cube):
    mean_dark = np.mean(dark_cube, axis=(0, 1))
    mean_white = np.mean(white_cube, axis=(0, 1))
    denominator = mean_white - mean_dark
    denominator[denominator == 0] = 1e-8
    reflectance_cube = (raw_cube - mean_dark) / denominator
    return np.clip(reflectance_cube, 0, 1)

def extrair_espectros_roi(reflectance_cube, roi_coords):
    centro = roi_coords["centro"]
    raio = roi_coords["raio"]
    altura, largura, _ = reflectance_cube.shape
    y_indices, x_indices = np.ogrid[:altura, :largura]
    dist_do_centro = np.sqrt((y_indices - centro[0])**2 + (x_indices - centro[1])**2)
    mask = dist_do_centro <= raio
    roi_pixels = reflectance_cube[mask]
    if roi_pixels.size == 0:
        raise ValueError(f"Coordenadas da ROI circular não encontraram pixels.")
    return roi_pixels

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    try:
        print(f"Carregando dados da amostra: '{amostra_info['nome']}'...")
        pasta_amostra = DATA_PATH / amostra_info["nome"]
        
        raw_cube = carregar_cubo_envi(pasta_amostra, amostra_info["arquivos"]["raw"])
        dark_cube = carregar_cubo_envi(pasta_amostra, amostra_info["arquivos"]["dark"])
        white_cube = carregar_cubo_envi(pasta_amostra, amostra_info["arquivos"]["white"])
        
        reflectance_cube = calibrar_para_refletancia(raw_cube, dark_cube, white_cube)
        matriz_pixels_roi = extrair_espectros_roi(reflectance_cube, amostra_info["roi_coords"])
        matriz_bandas_pixels = matriz_pixels_roi.T
        
        print(f"Matriz reorganizada criada com formato (bandas, pixels): {matriz_bandas_pixels.shape}")

        # --- ETAPA DE SALVAMENTO ---
        nome_base = Path(amostra_info["arquivos"]["raw"]).stem
        nome_arquivo_saida = f'matriz_bruta_{nome_base}.npy'
        caminho_arquivo_saida = SAVE_PATH / nome_arquivo_saida
        
        np.save(caminho_arquivo_saida, matriz_bandas_pixels)
        
        print("\n" + "="*50)
        print("✅ MATRIZ SALVA COM SUCESSO!")
        print(f"   Localização: {caminho_arquivo_saida}")
        print("="*50)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ ERRO: {e}")
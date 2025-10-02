import numpy as np
import spectral as sp
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# =============================================================================
# CONFIGURAÇÃO (🚨 EDITAR ESTA SEÇÃO)
# =============================================================================

# 1. Defina o caminho para a pasta que contém as pastas das suas amostras
DATA_PATH = Path('C:/Users/hugo/OneDrive/identificacao_bacteria/amostras_teste')

# 2. Liste suas amostras aqui. Para a ROI, defina o centro (y, x) e o raio em pixels.
AMOSTRAS_PARA_PROCESSAR = [
    {
        "nome": "351_1_240506-160920/capture",
        "classe": 1, # 1 para Resistente, 0 para Sensível
        "arquivos": {
            "raw": "351_1_240506-160920.raw",
            "dark": "DARKREF_351_1_240506-160920.raw",
            "white": "WHITEREF_351_1_240506-160920.raw"
        },
        "roi_coords": {
            "centro": (100, 150), # Coordenada (y, x) do centro do círculo
            "raio": 45            # Raio do círculo em pixels
        }
    },
    {
        "nome": "420_240506-160205/capture",
        "classe": 0,
        "arquivos": {
            "raw": "420_240506-160205.raw",
            "dark": "DARKREF_420_240506-160205.raw",
            "white": "WHITEREF_420_240506-160205.raw"
        },
        "roi_coords": {
            "centro": (95, 105),
            "raio": 20
        }
    },
    # Adicione mais amostras aqui...
]

# 3. Parâmetros de pré-processamento (baseado no seu plano de trabalho)
SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 2
BANDAS_PARA_REMOVER_INICIO = 5
BANDAS_PARA_REMOVER_FIM = 5


# =============================================================================
# FUNÇÕES AUXILIARES (NÃO PRECISA EDITAR)
# =============================================================================

def carregar_cubo_envi(pasta_amostra, nome_arquivo):
    """Carrega um arquivo ENVI e retorna como um array numpy."""
    caminho_raw = pasta_amostra / nome_arquivo
    caminho_hdr = caminho_raw.with_suffix('.hdr')
    
    if not caminho_hdr.exists():
        raise FileNotFoundError(f"Arquivo HDR não encontrado para {caminho_raw}")
        
    img = sp.envi.open(str(caminho_hdr), str(caminho_raw))
    return img.load()

def calibrar_para_refletancia(raw_cube, dark_cube, white_cube):
    """Converte dados brutos (DN) para refletância."""
    mean_dark = np.mean(dark_cube, axis=(0, 1))
    mean_white = np.mean(white_cube, axis=(0, 1))
    
    denominator = mean_white - mean_dark
    denominator[denominator == 0] = 1e-8
    
    reflectance_cube = (raw_cube - mean_dark) / denominator
    return np.clip(reflectance_cube, 0, 1)

def extrair_espectro_roi(reflectance_cube, roi_coords):
    """
    Extrai e calcula o espectro médio de uma ROI CIRCULAR.
    roi_coords deve ser um dicionário: {'centro': (y, x), 'raio': r}
    """
    centro = roi_coords["centro"]
    raio = roi_coords["raio"]
    
    # Cria uma grade de coordenadas para todos os pixels da imagem
    altura, largura, _ = reflectance_cube.shape
    y_indices, x_indices = np.ogrid[:altura, :largura]
    
    # Calcula a distância de cada pixel ao centro do círculo
    dist_do_centro = np.sqrt((y_indices - centro[0])**2 + (x_indices - centro[1])**2)
    
    # Cria uma máscara booleana: True para pixels dentro do círculo
    mask = dist_do_centro <= raio
    
    # Aplica a máscara para selecionar os pixels da ROI
    # O resultado é um array 2D: (numero_de_pixels_na_roi, numero_de_bandas)
    roi_pixels = reflectance_cube[mask]
    
    if roi_pixels.size == 0:
        raise ValueError(f"Coordenadas da ROI circular não encontraram pixels.")
        
    # Calcula a média ao longo do eixo dos pixels (axis=0)
    mean_spectrum = np.mean(roi_pixels, axis=0)
    return mean_spectrum

def preprocessar_espectro(spectrum):
    """Aplica a sequência completa de pré-processamento em um espectro."""
    processed = spectrum[BANDAS_PARA_REMOVER_INICIO:-BANDAS_PARA_REMOVER_FIM]
    processed = (processed - np.mean(processed)) / np.std(processed)
    processed = savgol_filter(processed, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER, deriv=1)
    processed = MinMaxScaler().fit_transform(processed.reshape(-1, 1)).ravel()
    return processed

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    matriz_espectros = []
    matriz_classes = []

    print(f"Iniciando processamento de {len(AMOSTRAS_PARA_PROCESSAR)} amostras...")

    for amostra_info in tqdm(AMOSTRAS_PARA_PROCESSAR, desc="Processando Amostras"):
        try:
            nome_amostra = amostra_info["nome"]
            pasta_amostra = DATA_PATH / nome_amostra
            
            raw_cube = carregar_cubo_envi(pasta_amostra, amostra_info["arquivos"]["raw"])
            dark_cube = carregar_cubo_envi(pasta_amostra, amostra_info["arquivos"]["dark"])
            white_cube = carregar_cubo_envi(pasta_amostra, amostra_info["arquivos"]["white"])
            
            reflectance_cube = calibrar_para_refletancia(raw_cube, dark_cube, white_cube)
            
            # A chamada da função continua a mesma, mas agora usa a lógica do círculo
            roi_spectrum = extrair_espectro_roi(reflectance_cube, amostra_info["roi_coords"])
            
            espectro_final = preprocessar_espectro(roi_spectrum)
            
            matriz_espectros.append(espectro_final)
            matriz_classes.append(amostra_info["classe"])

        except (FileNotFoundError, ValueError) as e:
            print(f"\n❌ ERRO ao processar '{nome_amostra}': {e}")
            print("   Pulando esta amostra.")

    X = np.array(matriz_espectros)
    y = np.array(matriz_classes)
    
    print("\n✅ Processamento concluído!")
    print(f"Formato da matriz de dados (X): {X.shape}")
    print(f"Formato da matriz de classes (y): {y.shape}")
    
    if len(matriz_espectros) > 0:
        plt.figure(figsize=(12, 7))
        plt.title("Efeito do Pré-processamento na Primeira Amostra (ROI Circular)")
        plt.xlabel("Bandas Espectrais (Índice)")
        plt.ylabel("Intensidade / Valor")
        
        info_primeira = AMOSTRAS_PARA_PROCESSAR[0]
        raw = carregar_cubo_envi(DATA_PATH / info_primeira["nome"], info_primeira["arquivos"]["raw"])
        dark = carregar_cubo_envi(DATA_PATH / info_primeira["nome"], info_primeira["arquivos"]["dark"])
        white = carregar_cubo_envi(DATA_PATH / info_primeira["nome"], info_primeira["arquivos"]["white"])
        reflectance = calibrar_para_refletancia(raw, dark, white)
        roi_spec_reflectance = extrair_espectro_roi(reflectance, info_primeira["roi_coords"])
        
        plt.plot(roi_spec_reflectance, label="1. Espectro de Refletância (após ROI)", alpha=0.7)
        plt.plot(X[0], label="2. Espectro Final (Após Pré-processamento)", linewidth=2)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
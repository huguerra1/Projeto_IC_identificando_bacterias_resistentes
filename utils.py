import cv2
import numpy as np
import spectral as sp
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew  # Necessário para o método 'estatisticas'
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from tqdm import tqdm

import config

# --- Configurações para suprimir avisos de bibliotecas ---
sp.settings.envi_support_nonlowercase_params = True
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# FUNÇÕES DE MANIPULAÇÃO DE DIRETÓRIOS E ARQUIVOS
# =============================================================================

def criar_diretorios_necessarios():
    """Cria todos os diretórios de saída se eles ainda não existirem."""
    print("Verificando e criando diretórios de saída...")
    try:
        config.RAW_MATRICES_PATH.mkdir(parents=True, exist_ok=True)
        config.ROI_VISUALS_PATH.mkdir(parents=True, exist_ok=True)
        config.DEBUG_PATH.mkdir(parents=True, exist_ok=True)
        config.PREPROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        config.RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Garante a pasta final de dados que os modelos usam
        Path("dados_gerados/02_dados_finais").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ ERRO ao criar diretórios: {e}")
        raise

def inferir_rotulo(nome_amostra: str) -> int:
    """Retorna 1 se o nome começa com 'ATCC' (Resistente), caso contrário, 0 (Sensível)."""
    return 1 if nome_amostra.upper().startswith("ATCC") else 0

# =============================================================================
# FUNÇÕES DE LEITURA E CALIBRAÇÃO DE DADOS HSI
# =============================================================================

def carregar_cubo_envi(pasta_amostra: Path, nome_arquivo: str) -> np.ndarray:
    """Carrega um arquivo de imagem HSI no formato ENVI (.raw/.hdr)."""
    caminho_raw = pasta_amostra / nome_arquivo
    caminho_hdr = caminho_raw.with_suffix('.hdr')
    if not caminho_hdr.exists():
        raise FileNotFoundError(f"Arquivo HDR não encontrado para {caminho_raw}")
    img = sp.envi.open(str(caminho_hdr), str(caminho_raw))
    return img.load()

def calibrar_para_refletancia(raw_cube: np.ndarray, dark_cube: np.ndarray, white_cube: np.ndarray) -> np.ndarray:
    """Converte DN para refletância usando média geométrica para evitar erro de dimensões."""
    mean_dark = np.mean(dark_cube, axis=(0, 1))
    mean_white = np.mean(white_cube, axis=(0, 1))
    denominator = mean_white - mean_dark
    denominator[denominator == 0] = 1e-8
    reflectance_cube = (raw_cube - mean_dark) / denominator
    return np.clip(reflectance_cube, 0, 1)

def get_rgb(cubo_hsi: np.ndarray, bands: list) -> np.ndarray:
    """Cria uma imagem RGB sintética normalizada para exibição."""
    rgb = cubo_hsi[..., bands].copy()
    for i in range(3):
        banda = rgb[..., i]
        min_val, max_val = np.min(banda), np.max(banda)
        if (max_val - min_val) > 0:
            rgb[..., i] = (banda - min_val) / (max_val - min_val)
        else:
            rgb[..., i] = 0
    return rgb

def encontrar_batoque_hough(cubo_hsi: np.ndarray, nome_amostra_base: str):
    """Detecta a ROI circular (batoque) usando a Transformada de Hough."""
    num_bands = cubo_hsi.shape[2]
    rgb_bands = [int(num_bands * 0.75), int(num_bands * 0.50), int(num_bands * 0.25)]
    
    rgb_image = get_rgb(cubo_hsi, rgb_bands)
    gray_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_blurred = cv2.GaussianBlur(gray_image, (7, 7), 2)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_contrasted = clahe.apply(gray_blurred)
    
    circles = cv2.HoughCircles(gray_contrasted, cv2.HOUGH_GRADIENT, 1.2, 100,
                               param1=100, param2=15, minRadius=50, maxRadius=200)

    if circles is None:
        debug_file_path = config.DEBUG_PATH / f"FALHA_CLAHE_{nome_amostra_base}_gray.png"
        cv2.imwrite(str(debug_file_path), gray_contrasted)
        raise ValueError(f"Nenhum círculo detectado para '{nome_amostra_base}'.")
        
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    x, y, r = circles[0][0]
    cv2.circle(mask, (int(x), int(y)), int(r), 1, thickness=-1)
    
    try:
        bgr_uint8_para_desenho = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.circle(bgr_uint8_para_desenho, (int(x), int(y)), int(r), (0, 255, 0), 2)
        save_name = config.ROI_VISUALS_PATH / f"DETECTADO_{nome_amostra_base}.png"
        cv2.imwrite(str(save_name), bgr_uint8_para_desenho)
    except Exception as e:
        pass

    return mask

# =============================================================================
# PRÉ-PROCESSAMENTO (ETAPA 03 DA METODOLOGIA)
# =============================================================================

def aplicar_preprocessamento(X: np.ndarray, metodo: str) -> np.ndarray:
    """Aplica uma técnica de pré-processamento selecionada."""
    print(f"Aplicando pré-processamento: '{metodo}'...")

    if metodo == 'savgol':
        return savgol_filter(X, window_length=config.SAVGOL_WINDOW,
                             polyorder=config.SAVGOL_POLYORDER, deriv=config.SAVGOL_DERIV, axis=1)
    elif metodo == 'snv':
        mean_spectra = np.mean(X, axis=1, keepdims=True)
        std_spectra = np.std(X, axis=1, keepdims=True)
        std_spectra[std_spectra == 0] = 1e-8
        return (X - mean_spectra) / std_spectra
    elif metodo == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(X)
    elif metodo == 'remocao_bandas':
        start = config.BANDS_TO_REMOVE_START
        end = X.shape[1] - config.BANDS_TO_REMOVE_END
        return X[:, start:end]
    elif metodo == 'nenhum':
        return X
    else:
        raise ValueError(f"Método de pré-processamento '{metodo}' não reconhecido.")

# =============================================================================
# EXTRAÇÃO E SELEÇÃO DE ATRIBUTOS (ETAPA 04 DA METODOLOGIA)
# =============================================================================

def selecionar_features(X: np.ndarray, y: np.ndarray, metodo: str) -> tuple[np.ndarray, np.ndarray]:
    """Aplica uma técnica de extração ou seleção de características."""
    n_features = X.shape[1]
    
    # Se o atributo config.N_FEATURES_TO_SELECT existir, usa ele, senão pega 50 por padrão
    n_to_select = getattr(config, 'N_FEATURES_TO_SELECT', 50)
    n_selecionar = min(n_to_select, n_features)
    
    print(f"Aplicando extração/seleção de features: '{metodo}'...")

    if metodo == 'anova':
        scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=0.0)
        indices_ordenados = np.argsort(scores)[::-1]
        indices_selecionados = indices_ordenados[:n_selecionar]
        X_selecionado = X[:, indices_selecionados]
        return X_selecionado, indices_selecionados
    
    elif metodo == 'mi':
        scores = mutual_info_classif(X, y)
        indices_ordenados = np.argsort(scores)[::-1]
        indices_selecionados = indices_ordenados[:n_selecionar]
        X_selecionado = X[:, indices_selecionados]
        return X_selecionado, indices_selecionados
        
    elif metodo == 'estatisticas':
        # Cálculo das 4 propriedades estatísticas espectrais descritas na sua metodologia
        mean_vals = np.mean(X, axis=1, keepdims=True)
        std_vals = np.std(X, axis=1, keepdims=True)
        kurt_vals = kurtosis(X, axis=1, fisher=True).reshape(-1, 1)
        skew_vals = skew(X, axis=1).reshape(-1, 1)
        
        X_stats = np.hstack([X, mean_vals, std_vals, kurt_vals, skew_vals])
        indices_totais = np.arange(X_stats.shape[1])
        return X_stats, indices_totais

    elif metodo == 'nenhum':
        return X, np.arange(n_features)
        
    else:
        raise ValueError(f"Método '{metodo}' não reconhecido.")
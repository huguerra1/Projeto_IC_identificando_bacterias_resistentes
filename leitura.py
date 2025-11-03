import numpy as np
import spectral as sp
import cv2  # Biblioteca do PDF
from pathlib import Path
from tqdm import tqdm
import warnings

# --- Configurações para suprimir avisos ---
sp.settings.envi_support_nonlowercase_params = True
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

DATA_PATH = Path('C:/Users/hugo/OneDrive/identificacao_bacteria/hsi_original/hsi_original')
SAVE_PATH = Path('./matrizes_salvas_BATOQUE')
DEBUG_PATH = Path('./debug_imagens_falhas') # Pasta para imagens de FALHA
SAVE_PATH_VISUALS = Path('./salvas_visuais_batoque') # <-- NOVO: Pasta para imagens de SUCESSO

SAVE_PATH.mkdir(exist_ok=True)
DEBUG_PATH.mkdir(exist_ok=True)
SAVE_PATH_VISUALS.mkdir(exist_ok=True) # <-- NOVO: Cria a pasta

# =============================================================================
# FUNÇÕES AUXILIARES
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

def get_rgb(cubo_hsi, bands):
    """Cria uma imagem RGB (como um array NumPy) a partir de um cubo HSI."""
    rgb = cubo_hsi[..., bands].copy()
    for i in range(3):
        banda = rgb[..., i]
        min_val = np.min(banda)
        max_val = np.max(banda)
        if (max_val - min_val) > 0:
            rgb[..., i] = (banda - min_val) / (max_val - min_val)
        else:
            rgb[..., i] = 0
    return rgb


def encontrar_batoque_hough(cubo_hsi, nome_amostra_base):
    """
    Encontra o batoque circular usando a Transformada de Hough.
    Usa bandas dinâmicas, salva imagem de debug e aplica CLAHE para contraste.
    """
    
    # --- Usa bandas dinâmicas ---
    num_bands = cubo_hsi.shape[2]
    band_r = int(num_bands * 0.75)
    band_g = int(num_bands * 0.50)
    band_b = int(num_bands * 0.25)
    rgb_bands = [band_r, band_g, band_b]
    
    print(f"   ...usando bandas DINÂMICAS {rgb_bands} para gerar imagem RGB sintética...")
    
    # 1. Geração de imagem RGB sintética
    rgb = get_rgb(cubo_hsi, rgb_bands) 
    
    # Converte para escala de cinza 8-bit para o OpenCV
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Suavização para reduzir ruído (ainda importante)
    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    
    # Aumento de Contraste com CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_contrasted = clahe.apply(gray_blurred)
    
    # 2. Segmentação automática com Transformada de Hough
    circles = cv2.HoughCircles(gray_contrasted, cv2.HOUGH_GRADIENT, 1.2, 100,
                               param1=100, param2=15, 
                               minRadius=50, maxRadius=200) 

    if circles is None:
        # Salva a imagem COM CONTRASTE que falhou
        debug_file_path = DEBUG_PATH / f"FALHA_CLAHE_{nome_amostra_base}_gray.png"
        cv2.imwrite(str(debug_file_path), gray_contrasted) # Salva a imagem processada pelo CLAHE
        # Levanta o erro informando sobre o arquivo de debug
        raise ValueError(f"Nenhum círculo detectado (param2=15). Imagem de debug salva em: {debug_file_path}")
        
    # 3. Criação de máscara binária
    mask = np.zeros(gray.shape, dtype=np.uint8)
    x, y, r = circles[0][0]
    
    # --- NOVO: SALVAR IMAGEM DE VISUALIZAÇÃO (APÓS SUCESSO) ---
    try:
        # Converte a imagem RGB (float 0-1) para BGR (uint8 0-255) para o OpenCV
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        bgr_uint8_para_desenho = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        
        # Desenha o círculo (verde, com espessura 2)
        cv2.circle(bgr_uint8_para_desenho, (int(x), int(y)), int(r), (0, 255, 0), 2)
        
        # Salva a imagem
        save_name = SAVE_PATH_VISUALS / f"DETECTADO_{nome_amostra_base}.png"
        cv2.imwrite(str(save_name), bgr_uint8_para_desenho)
        print(f"   ...Visualização do círculo salva em: {save_name}")
    except Exception as e:
        print(f"   AVISO: Falha ao salvar imagem de visualização: {e}")
    # --- FIM DA NOVA SEÇÃO ---

    cv2.circle(mask, (int(x), int(y)), int(r), 1, thickness=-1)
    
    return mask, (int(x), int(y), int(r))

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("Iniciando pipeline de processamento automático (com Hough/Batoque e Debug)...")
    
    pastas_para_processar = [p for p in DATA_PATH.iterdir() if p.is_dir()]
    print(f"Encontradas {len(pastas_para_processar)} pastas de amostras para processar.")

    for pasta_amostra in tqdm(pastas_para_processar, desc="Processando Amostras"):
        nome_amostra_base = pasta_amostra.name
        
        try:
            print(f"\nCarregando dados da amostra: '{nome_amostra_base}'...")
            
            pasta_capture = pasta_amostra / "capture"
            arquivos = {
                "raw": f"{nome_amostra_base}.raw",
                "dark": f"DARKREF_{nome_amostra_base}.raw",
                "white": f"WHITEREF_{nome_amostra_base}.raw"
            }
            
            # --- Carrega e Calibra ---
            raw_cube = carregar_cubo_envi(pasta_capture, arquivos["raw"])
            dark_cube = carregar_cubo_envi(pasta_capture, arquivos["dark"])
            white_cube = carregar_cubo_envi(pasta_capture, arquivos["white"])
            reflectance_cube = calibrar_para_refletancia(raw_cube, dark_cube, white_cube)
            
            # --- ETAPA DE DETECÇÃO AUTOMÁTICA (DO PDF) ---
            print("   Detectando batoque com Transformada de Hough...")
            # Passa o nome da amostra para a função de debug
            mask_2d, (x, y, r) = encontrar_batoque_hough(reflectance_cube, nome_amostra_base)
            print(f"   Batoque encontrado em (x={x}, y={y}) com raio={r}")
            
            # 4. Aplicação da máscara e Recorte da ROI
            mask_3d = np.repeat(mask_2d[:, :, np.newaxis], reflectance_cube.shape[2], axis=2)
            matriz_pixels_roi = reflectance_cube[mask_2d == 1]
            matriz_bandas_pixels = matriz_pixels_roi.T 
            
            print(f"Matriz reorganizada (Batoque) criada com formato (bandas, pixels): {matriz_bandas_pixels.shape}")

            # --- ETAPA DE SALVAMENTO ---
            nome_arquivo_saida = f'matriz_bruta_BATOQUE_{nome_amostra_base}.npy'
            caminho_arquivo_saida = SAVE_PATH / nome_arquivo_saida
            np.save(caminho_arquivo_saida, matriz_bandas_pixels)
            
            print(f"✅ MATRIZ SALVA COM SUCESSO! Local: {caminho_arquivo_saida}")

        except (FileNotFoundError, ValueError, TypeError) as e:
            # O erro agora conterá a informação do arquivo de debug
            print(f"\n❌ ERRO ao processar '{nome_amostra_base}': {e}. Pulando esta amostra.")
    
    print("\n" + "="*50)
    print("Processamento de todas as amostras concluído!")
    print("="*50)
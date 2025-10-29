import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings

# --- Configurações para suprimir avisos ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURAÇÃO (🚨 EDITAR ESTA SEÇÃO)
# =============================================================================

# 1. Defina o caminho para a pasta onde as matrizes BRUTAS estão salvas
LOAD_PATH = Path('./matrizes_salvas')
# 2. Defina o caminho para a pasta onde as matrizes PROCESSADAS serão salvas
SAVE_PATH = Path('./matrizes_processadas')
SAVE_PATH.mkdir(exist_ok=True) # Cria a pasta se ela não existir

# 3. Parâmetros de pré-processamento
SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 2
BANDAS_PARA_REMOVER_INICIO = 15
BANDAS_PARA_REMOVER_FIM = 15

# =============================================================================
# FUNÇÃO DE PRÉ-PROCESSAMENTO (NÃO PRECISA EDITAR)
# =============================================================================

def preprocessar_espectros(spectra_array):
    """Aplica a sequência de pré-processamento em cada espectro (linha) do array."""
    processed_list = []
    # tqdm adiciona uma barra de progresso, útil para muitos pixels
    for spectrum in tqdm(spectra_array, desc="Processando pixels", leave=False):
        processed = spectrum[BANDAS_PARA_REMOVER_INICIO:-BANDAS_PARA_REMOVER_FIM]
        processed = (processed - np.mean(processed)) / np.std(processed)
        processed = savgol_filter(processed, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER, deriv=1)
        processed = MinMaxScaler().fit_transform(processed.reshape(-1, 1)).ravel()
        processed_list.append(processed)
    return np.array(processed_list)

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    
    # --- ETAPA 1: Encontrar todos os arquivos .npy brutos ---
    arquivos_para_processar = list(LOAD_PATH.glob('matriz_bruta_*.npy'))
    
    if not arquivos_para_processar:
        print(f"❌ ERRO: Nenhum arquivo 'matriz_bruta_*.npy' encontrado em '{LOAD_PATH}'")
    else:
        print(f"Encontrados {len(arquivos_para_processar)} arquivos para processar.")
        
        # --- ETAPA 2: Loop para processar cada arquivo encontrado ---
        for caminho_arquivo_entrada in tqdm(arquivos_para_processar, desc="Processando Matrizes (Total)"):
            try:
                print(f"\nCarregando a matriz de '{caminho_arquivo_entrada.name}'...")
                matriz_bandas_pixels = np.load(caminho_arquivo_entrada)
                print(f"Matriz carregada com formato (bandas, pixels): {matriz_bandas_pixels.shape}")

                # --- ETAPA 3: Transpor para o formato (pixels, bandas) ---
                matriz_pixels_bandas = matriz_bandas_pixels.T
                print(f"Matriz transposta para o formato (pixels, bandas): {matriz_pixels_bandas.shape}")

                # --- ETAPA 4: Aplicar o pré-processamento ---
                print("Iniciando pré-processamento...")
                X_processado = preprocessar_espectros(matriz_pixels_bandas)
                print("Pré-processamento concluído!")

                # --- ETAPA 5: Salvar a matriz processada ---
                nome_base = caminho_arquivo_entrada.stem # Pega o nome do arquivo sem .npy
                novo_nome = nome_base.replace('matriz_bruta_', 'matriz_processada_') + '.npy'
                caminho_arquivo_saida = SAVE_PATH / novo_nome
                
                np.save(caminho_arquivo_saida, X_processado)
                print(f"✅ Matriz processada salva em: {caminho_arquivo_saida}")
                print(f"   Formato final: {X_processado.shape}")
                print("-"*50)

            except Exception as e:
                print(f"❌ ERRO ao processar o arquivo {caminho_arquivo_entrada.name}: {e}")
                
        print("\n" + "="*50)
        print("✅ Processamento de todas as matrizes concluído!")
        print(f"Arquivos processados salvos em '{SAVE_PATH}'")
        print("="*50)
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# =============================================================================
# CONFIGURAÇÃO (🚨 EDITAR ESTA SEÇÃO)
# =============================================================================

# 1. Defina o caminho para a pasta onde a matriz foi salva
LOAD_PATH = Path('./matrizes_salvas')
# 2. Defina o nome do arquivo .npy que você quer processar
NOME_ARQUIVO_ENTRADA = 'matriz_bruta_351_1_240506-160920.npy'

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
    for spectrum in tqdm(spectra_array, desc="Processando pixels"):
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
    caminho_arquivo_entrada = LOAD_PATH / NOME_ARQUIVO_ENTRADA
    
    if not caminho_arquivo_entrada.exists():
        print(f"❌ ERRO: Arquivo de entrada não encontrado em '{caminho_arquivo_entrada}'")
    else:
        # --- ETAPA 1: Carregar a matriz salva ---
        print(f"Carregando a matriz de '{caminho_arquivo_entrada}'...")
        matriz_bandas_pixels = np.load(caminho_arquivo_entrada)
        print(f"Matriz carregada com formato (bandas, pixels): {matriz_bandas_pixels.shape}")

        # --- ETAPA 2: Transpor para o formato (pixels, bandas) para o processamento ---
        # É mais eficiente processar linhas (pixels) do que colunas
        matriz_pixels_bandas = matriz_bandas_pixels.T
        print(f"Matriz transposta para o formato (pixels, bandas): {matriz_pixels_bandas.shape}")

        # --- ETAPA 3: Aplicar o pré-processamento ---
        print("\nIniciando pré-processamento...")
        X_processado = preprocessar_espectros(matriz_pixels_bandas)

        print("\n" + "="*50)
        print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO!")
        print(f"   Formato final da matriz processada (pixels, bandas_processadas): {X_processado.shape}")
        print("="*50)

        # Agora, a variável 'X_processado' contém seus dados limpos e prontos para o modelo.
        # Opcional: Salvar a matriz processada
        # nome_saida_proc = f'matriz_processada_{Path(NOME_ARQUIVO_ENTRADA).stem}.npy'
        # np.save(LOAD_PATH / nome_saida_proc, X_processado)
        # print(f"Matriz processada salva em: {LOAD_PATH / nome_saida_proc}")
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib.pyplot as plt

# --- Configurações para suprimir avisos ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURAÇÃO (🚨 EDITAR ESTA SEÇÃO)
# =============================================================================

# 1. Defina o caminho para a pasta onde as matrizes BRUTAS .npy estão salvas
LOAD_PATH = Path('./matrizes_salvas')

# 2. Defina o nome do ARQUIVO .NPY que você quer carregar
NOME_MATRIZ_SALVA = 'matriz_bruta_351_1_240506-160920.npy'

# 3. Defina o ÍNDICE DO PIXEL que você quer analisar
PIXEL_INDEX_PARA_ANALISAR = 0 

# 4. Defina o intervalo de comprimento de onda (para o eixo X)
WAVELENGTH_START = 900
WAVELENGTH_END = 2500

# 5. Parâmetros de pré-processamento
SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 2
BANDAS_PARA_REMOVER_INICIO = 15
BANDAS_PARA_REMOVER_FIM = 15

# =============================================================================
# FUNÇÃO AUXILIAR (MODIFICADA)
# =============================================================================

def preprocessar_etapas(spectrum):
    """
    Aplica a sequência completa de pré-processamento e retorna
    os resultados de cada etapa intermediária.
    """
    
    # Etapa 1: Remoção de Bandas Ruidosas
    processed_1_corte = spectrum[BANDAS_PARA_REMOVER_INICIO:-BANDAS_PARA_REMOVER_FIM]
    
    # Etapa 2: SNV (Standard Normal Variate)
    std_dev = np.std(processed_1_corte)
    if std_dev == 0: std_dev = 1e-8 # Evita divisão por zero
    processed_2_snv = (processed_1_corte - np.mean(processed_1_corte)) / std_dev
    
    # Etapa 3: Filtro Savitzky-Golay (Derivada)
    processed_3_savgol = savgol_filter(processed_2_snv, 
                                     window_length=SAVGOL_WINDOW, 
                                     polyorder=SAVGOL_POLYORDER, 
                                     deriv=1)
    
    # Etapa 4: Normalização Min-Max
    processed_4_final = MinMaxScaler().fit_transform(processed_3_savgol.reshape(-1, 1)).ravel()
    
    # Retorna o resultado de todas as etapas
    return processed_1_corte, processed_2_snv, processed_3_savgol, processed_4_final

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    try:
        # --- 1. Carregar Matriz .npy ---
        caminho_matriz_salva = LOAD_PATH / NOME_MATRIZ_SALVA
        if not caminho_matriz_salva.exists():
            raise FileNotFoundError(f"Arquivo de Matriz Salva não encontrado: {caminho_matriz_salva}")
            
        print(f"Carregando matriz salva: '{NOME_MATRIZ_SALVA}'...")
        matriz_bandas_pixels = np.load(caminho_matriz_salva)
        print(f"Matriz carregada com formato (bandas, pixels): {matriz_bandas_pixels.shape}")

        # --- 2. Gerar Comprimentos de Onda (Eixo X) ---
        num_bands_original = matriz_bandas_pixels.shape[0]
        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, num_bands_original)
        
        # Gera o eixo X cortado para os gráficos processados
        wavelengths_cortados = wavelengths[BANDAS_PARA_REMOVER_INICIO:-BANDAS_PARA_REMOVER_FIM]
        
        print(f"Eixo X gerado: {len(wavelengths)} bandas de {wavelengths[0]:.2f}nm a {wavelengths[-1]:.2f}nm")

        # --- 3. Extrair o Pixel de Interesse ---
        espectro_antes = matriz_bandas_pixels[:, PIXEL_INDEX_PARA_ANALISAR]
        print(f"Espectro 'Antes' extraído do pixel de índice {PIXEL_INDEX_PARA_ANALISAR}")

        # --- 4. Pré-processar o Espectro (obtendo todas as etapas) ---
        e_corte, e_snv, e_savgol, e_final = preprocessar_etapas(espectro_antes)
        print("Pré-processamento do espectro concluído.")

        # --- 5. Plotar os Gráficos ---
        print("Gerando gráfico de comparação etapa por etapa...")
        
        # Cria uma figura com 4 subplots verticais
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            nrows=4, 
            ncols=1, 
            figsize=(12, 18) # Figura alta para caber tudo
        )
        fig.suptitle(f"Análise de Pixel (Índice {PIXEL_INDEX_PARA_ANALISAR}) - Etapa por Etapa", fontsize=16, y=1.02)

        # Gráfico 1: Original (Refletância Bruta)
        ax1.plot(wavelengths, espectro_antes, color='steelblue')
        ax1.set_title("1. Original (Refletância Bruta)")
   
        ax1.set_ylabel("Refletância")
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Gráfico 2: Após SNV (note que já inclui o corte de bandas)
        ax2.plot(wavelengths_cortados, e_snv, color='green')
        ax2.set_title("2. Após Corte de Bandas + SNV")
    
        ax2.set_ylabel("Valor (Pós-SNV)")
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Gráfico 3: Após Savitzky-Golay
        ax3.plot(wavelengths_cortados, e_savgol, color='red')
        ax3.set_title("3. Após Savitzky-Golay (Derivada)")

        ax3.set_ylabel("Valor (Derivada)")
        ax3.grid(True, linestyle='--', alpha=0.5)

        # Gráfico 4: Final (Min-Max)
        ax4.plot(wavelengths_cortados, e_final, color='darkorange')
        ax4.set_title("4. Espectro Final (Após Min-Max)")

        ax4.set_ylabel("Valor (Normalizado 0-1)")
        ax4.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        plt.show()

    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"\n❌ ERRO: {e}")
        print("   Verifique os caminhos, nomes de arquivos e se o índice do pixel está dentro dos limites da matriz.")
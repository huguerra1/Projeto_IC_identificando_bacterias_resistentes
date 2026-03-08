from pathlib import Path

# =============================================================================
# CONFIGURAÇÃO GERAL DO PIPELINE
# =============================================================================

# --- 1. DIRETÓRIOS BASE ---
# Aponta para a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent

# Diretório para onde todos os dados processados e resultados irão.
# Isso centraliza a saída, facilitando o acesso e a limpeza.
OUTPUT_DIR = BASE_DIR / 'dados_gerados'

# --- 2. SUBDIRETÓRIOS (serão criados automaticamente) ---

# Etapa 1: Leitura e Extração da ROI
HSI_ORIGINAL_PATH = BASE_DIR / 'hsi_original' / 'hsi_original'
RAW_MATRICES_PATH = OUTPUT_DIR / '01_matrizes_brutas'
ROI_VISUALS_PATH = OUTPUT_DIR / '00_visualizacoes_roi'
DEBUG_PATH = OUTPUT_DIR / '00_debug_leitura'

# Etapa 2: Pré-processamento e Seleção de Features
PREPROCESSED_PATH = OUTPUT_DIR / '02_dados_finais'

# Etapa 3: Treinamento e Resultados
RESULTS_PATH = OUTPUT_DIR / '03_resultados_treinamento'
MODELS_PATH = RESULTS_PATH / 'modelos'


# --- 3. PARÂMETROS DE PRÉ-PROCESSAMENTO ---

# Parâmetros para o filtro Savitzky-Golay
SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 2
SAVGOL_DERIV = 1

# Parâmetros para remoção de bandas ruidosas nas extremidades
BANDS_TO_REMOVE_START = 15
BANDS_TO_REMOVE_END = 15


# --- 4. PARÂMETROS DE SELEÇÃO DE FEATURES ---

# Número de bandas/features a serem selecionadas pelos métodos
N_FEATURES_TO_SELECT = 100


# --- 5. PARÂMETROS DE TREINAMENTO ---

# Proporção do conjunto de dados a ser usada para teste
TEST_SET_SIZE = 0.3

# Semente aleatória para garantir a reprodutibilidade na divisão de dados e no modelo
RANDOM_STATE = 42

# Hiperparâmetros para a busca em grade (GridSearch) do Decision Tree
GRID_SEARCH_PARAMS = {
    'criterion': ['gini'],           # Teste apenas um critério por enquanto
    'max_depth': [None, 10, 20],     # Reduzido de 5 para 3 opções
    'min_samples_leaf': [1, 2],      # Reduzido de 3 para 2 opções
    'ccp_alpha': [0.0, 0.001]        # Reduzido de 4 para 2 opções
}
GRID_SEARCH_XGB = {
    'n_estimators': [50, 100],        # Número de árvores no conjunto
    'learning_rate': [0.1],           # Velocidade de aprendizado (0.1 é o padrão robusto)
    'max_depth': [3, 6],              # Profundidade das árvores individuais
    'subsample': [0.8]                # Porcentagem de dados usada em cada árvore
}

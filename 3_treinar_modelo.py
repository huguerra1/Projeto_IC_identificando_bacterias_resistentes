import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # Para salvar o modelo

# Importa os módulos centralizados de configuração e utilidades
import config
import utils

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def load_dataset(preproc_method: str, feature_method: str) -> tuple:
    """
    Carrega os conjuntos de dados X e y processados pela Etapa 2.
    """
    base_filename = f"{preproc_method}_{feature_method}"
    x_path = config.PREPROCESSED_PATH / f"X_{base_filename}.npy"
    y_path = config.PREPROCESSED_PATH / f"y_{base_filename}.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Arquivos de dados não encontrados para a combinação '{base_filename}'.\n"
            f"Verificado em: '{config.PREPROCESSED_PATH}'.\n"
            "Certifique-se de que a Etapa 2 (processar_dados.py) foi executada para esta combinação."
        )

    print(f"Carregando dados processados: {base_filename}")
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"Dados carregados. Shape X: {X.shape}, Shape y: {y.shape}")
    
    return X, y

def save_results(report: str, preproc_method: str, feature_method: str):
    """ Salva o relatório de texto com os resultados. """
    filename = f"relatorio_{preproc_method}_{feature_method}.txt"
    filepath = config.RESULTS_PATH / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Relatório de resultados salvo em: {filepath}")

def plot_and_save_tree(model, preproc_method: str, feature_method: str):
    """ Plota e salva a imagem da árvore de decisão. """
    filename = f"arvore_{preproc_method}_{feature_method}.png"
    filepath = config.RESULTS_PATH / filename
    
    plt.figure(figsize=(30, 15))
    plot_tree(
        model,
        filled=True,
        rounded=True,
        class_names=['Não Resistente', 'Resistente'], # Assumindo 0 e 1
        feature_names=[f"Feature_{i}" for i in range(model.n_features_in_)]
    )
    plt.title(f"Árvore de Decisão: {preproc_method} + {feature_method}", fontsize=20)
    
    print(f"Salvando visualização da árvore em: {filepath}")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, preproc_method: str, feature_method: str):
    """ Plota e salva a imagem da matriz de confusão. """
    filename = f"matriz_confusao_{preproc_method}_{feature_method}.png"
    filepath = config.RESULTS_PATH / filename
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Resistente', 'Resistente'],
                yticklabels=['Não Resistente', 'Resistente'])
    plt.title(f'Matriz de Confusão: {preproc_method} + {feature_method}', fontsize=16)
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    
    print(f"Salvando matriz de confusão em: {filepath}")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# =============================================================================

def main(preproc_method: str, feature_method: str):
    """
    Executa o pipeline completo da Etapa 3: treinamento, avaliação e
    salvamento de resultados.
    """
    print("="*70)
    print("INICIANDO ETAPA 3: Treinamento e Avaliação do Modelo")
    print(f" - Usando dados: {preproc_method} + {feature_method}")
    print("="*70)

    try:
        # --- 1. Carregar os dados ---
        X, y = load_dataset(preproc_method, feature_method)

        # --- 2. Divisão em Treino e Teste ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SET_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
        if len(X_train) > 50000:
            print(f"Subamostrando treino de {len(X_train)} para 50.000 pixels para velocidade...")
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=50000, stratify=y_train, random_state=config.RANDOM_STATE
            )
        print(f"Dados divididos em {len(X_train)} amostras de treino e {len(X_test)} de teste.")

        # --- 3. Configuração e execução do GridSearchCV ---
        print("\nIniciando busca de hiperparâmetros com GridSearchCV (k-fold=5)...")
        grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            param_grid=config.GRID_SEARCH_PARAMS,
            cv=5,         
            scoring='accuracy',
            n_jobs=-1,     # Continua usando todos os núcleos
            verbose=3      # AGORA você verá o terminal se mexendo sem parar!
        )
        grid_search.fit(X_train, y_train)

        # --- 4. Extração do melhor modelo e avaliação ---
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # --- 5. Geração do relatório de resultados ---
        report = f"""
# ====================================================================
# RELATÓRIO DE TREINAMENTO E AVALIAÇÃO
# ====================================================================

# Combinação de Métodos:
# - Pré-processamento: {preproc_method}
# - Seleção de Features: {feature_method}

# --------------------------------------------------------------------
# Resultados do GridSearchCV
# --------------------------------------------------------------------
Melhores parâmetros encontrados:
{grid_search.best_params_}

Melhor score (acurácia) na validação cruzada:
{grid_search.best_score_:.4f}

# --------------------------------------------------------------------
# Avaliação no Conjunto de Teste
# --------------------------------------------------------------------
Acurácia no teste: {accuracy:.4f}

Relatório de Classificação:
{class_report}
"""
        print(report)

        # --- 6. Salvamento dos resultados e do modelo ---
        base_filename = f"{preproc_method}_{feature_method}"
        
        # Salva o relatório de texto
        save_results(report, preproc_method, feature_method)
        
        # Salva o modelo treinado
        model_path = config.MODELS_PATH / f"modelo_{base_filename}.joblib"
        joblib.dump(best_model, model_path)
        print(f"Modelo treinado salvo em: {model_path}")
        
        # Salva as visualizações gráficas
        plot_and_save_tree(best_model, preproc_method, feature_method)
        plot_and_save_confusion_matrix(y_test, y_pred, preproc_method, feature_method)
        
        print("\n" + "="*70)
        print(f"✅ Etapa 3 concluída para '{base_filename}'!")
        print(f"Resultados salvos em: '{config.RESULTS_PATH}'")
        print(f"Modelos salvos em: '{config.MODELS_PATH}'")
        print("="*70)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ ERRO durante a Etapa 3: {e}")

# =============================================================================
# PONTO DE ENTRADA DO SCRIPT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script da Etapa 3: Treina e avalia um modelo de Árvore de Decisão."
    )
    
    # Argumentos para selecionar a combinação de dados
    parser.add_argument(
        "--preproc", 
        type=str, 
        required=True,
        help="Método de pré-processamento usado (ex: 'savgol', 'snv', 'nenhum')."
    )
    parser.add_argument(
        "--feature", 
        type=str, 
        required=True, 
        help="Método de seleção de features usado (ex: 'anova', 'mi', 'nenhum')."
    )
    
    args = parser.parse_args()

    # Garante que os diretórios existam antes de executar
    utils.criar_diretorios_necessarios()
    
    # Chama a função principal com os argumentos fornecidos
    main(args.preproc, args.feature)

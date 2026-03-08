import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

import config
import utils

def main(preproc_method: str, feature_method: str):
    print("="*70)
    print("INICIANDO ETAPA 3 (B): Treinamento com Gradient Boosting (XGBoost)")
    print(f" - Usando dados: {preproc_method} + {feature_method}")
    print("="*70)

    try:
        # 1. Carregar os dados processados na Etapa 2
        base_filename = f"{preproc_method}_{feature_method}"
        X = np.load(config.PREPROCESSED_PATH / f"X_{base_filename}.npy")
        y = np.load(config.PREPROCESSED_PATH / f"y_{base_filename}.npy")

        # 2. Divisão Estratificada (70% Treino / 30% Teste) [cite: 62]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SET_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )

        # 2. SUBAMOSTRAGEM (O segredo da velocidade)
        if len(X_train) > 50000:
            print(f"Subamostrando treino para 50.000 pixels...")
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=50000, stratify=y_train, random_state=config.RANDOM_STATE
            )

        # 3. Configuração do XGBoost e GridSearchCV 
        # O XGBoost é um conjunto de árvores que corrige erros sequencialmente
        model_xgb = XGBClassifier(
            random_state=config.RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # Hiperparâmetros específicos para Boosting
        param_grid = {
            'n_estimators': [50, 100, 200], # Número de árvores
            'learning_rate': [0.01, 0.1, 0.2], # Taxa de aprendizado
            'max_depth': [3, 5, 7],           # Profundidade de cada árvore
            'subsample': [0.8, 1.0]           # Fração de pixels usada por árvore
        }

        grid_search = GridSearchCV(
            estimator=XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss'),
            param_grid=config.GRID_SEARCH_XGB,
            cv=5,              # Reduzido para 3 folds
            scoring='accuracy',
            n_jobs=-1,         # Usa todos os núcleos da sua CPU
            verbose=3          # Para você ver o progresso em tempo real
        )

        print("Treinando modelos e ajustando hiperparâmetros...")
        grid_search.fit(X_train, y_train)

        # 4. Avaliação e Métricas [cite: 66]
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        print(f"\nMelhores Parâmetros: {grid_search.best_params_}")
        print(f"Acurácia no Teste: {accuracy_score(y_test, y_pred):.4f}")
        print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

        # 5. Salvar Resultados e Gráficos [cite: 67]
        # Matriz de Confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Sensível', 'MRSA (ATCC)'], yticklabels=['Sensível', 'MRSA (ATCC)'])
        plt.title(f'XGBoost: {preproc_method} + {feature_method}')
        plt.savefig(config.RESULTS_PATH / f"confusao_xgboost_{base_filename}.png")
        
        # Salvar Modelo
        joblib.dump(best_model, config.MODELS_PATH / f"xgboost_{base_filename}.joblib")
        print(f"✅ Sucesso! Resultados salvos em {config.RESULTS_PATH}")

    except Exception as e:
        print(f"❌ ERRO: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preproc", type=str, required=True)
    parser.add_argument("--feature", type=str, required=True)
    args = parser.parse_args()
    main(args.preproc, args.feature)
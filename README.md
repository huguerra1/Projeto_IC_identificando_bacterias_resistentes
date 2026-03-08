# Identificação de Bactérias Resistentes via Imagens Hiperespectrais (HSI)

Este repositório contém o pipeline de processamento e classificação desenvolvido para o projeto de Iniciação Científica (IC). O objetivo é identificar perfis de resistência bacteriana utilizando aprendizado de máquina.

## 📂 Estrutura do Projeto

O código está organizado de forma sequencial para facilitar a reprodução dos experimentos:

1.  **`1_extrair_dados_brutos.py`**: Conversão e extração de dados das imagens `.hsi`.
2.  **`2_processar_e_selecionar.py`**: Limpeza, normalização e feature selection.
3.  **`3_treinar_xgboost.py`**: Treinamento do modelo de Gradient Boosting para classificação.
4.  **`4_analisar_amostra.py`**: Script para inferência e teste em novas amostras.

### Scripts de Suporte:
* `executar_experimentos.py`: Automação de bateria de testes.
* `gerar_tabela_resultados.py`: Geração de métricas (Acurácia, F1-Score, Matriz de Confusão).
* `config.py`: Definição de caminhos e hiperparâmetros.
* `utils.py`: Funções auxiliares (I/O e processamento).

## 🛠️ Tecnologias e Dependências

* **Python 3.x**
* **XGBoost**: Algoritmo de classificação principal.
* **Scikit-Learn**: Divisão de dados e métricas.
* **Pandas/NumPy**: Manipulação de dados hiperespectrais.

Para instalar as dependências:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib
⚠️ Observação sobre os Dados
Por questões de armazenamento, as pastas hsi_original/ (dados brutos) e dados_gerados/ (modelos treinados e saídas) não estão incluídas neste repositório devido ao grande volume de dados (1GB+).

Para rodar os scripts, certifique-se de que os dados brutos estejam na pasta local conforme configurado em config.py.

Desenvolvido por Hugo Mourão Projeto de Iniciação Científica - 2026.

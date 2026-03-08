import pandas as pd
from pathlib import Path
import re

PASTA_RESULTADOS = Path("dados_gerados/03_resultados_treinamento")

def extrair_metricas(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()

    # Busca acurácia (formato 0.9351 ou 0.93)
    acc_match = re.search(r"Acurácia no [Tt]este:\s*(0\.\d+)|accuracy\s+(0\.\d+)", conteudo)
    # Busca métricas da classe 1 (Resistente)
    linha_classe_1_match = re.search(r"1\s+(0\.\d+)\s+(0\.\d+)\s+(0\.\d+)\s+\d+", conteudo)

    if acc_match and linha_classe_1_match:
        acuracia = float(acc_match.group(1) or acc_match.group(2))
        return {
            "Acurácia": acuracia,
            "Precisão": float(linha_classe_1_match.group(1)),
            "Recall": float(linha_classe_1_match.group(2)),
            "F1-Score": float(linha_classe_1_match.group(3))
        }
    return None

def compilar_resultados():
    print("🔍 Gerando Ranking Comparativo (XGBoost vs Árvore)...")
    dados = []
    
    for arquivo in PASTA_RESULTADOS.glob("*.txt"):
        metricas = extrair_metricas(arquivo)
        if metricas:
            nome_limpo = arquivo.stem.lower()
            modelo = "XGBoost" if "xgboost" in nome_limpo else "Árvore de Decisão"
            
            metodos = nome_limpo.replace("relatorio_", "").replace("xgboost_", "").replace("_", " ").upper()
            
            dados.append({
                "Modelo": modelo,
                "Métodos": metodos,
                "Acurácia (%)": round(metricas["Acurácia"] * 100, 2),
                "Precisão ATCC": metricas["Precisão"],
                "Recall ATCC": metricas["Recall"],
                "F1-Score ATCC": metricas["F1-Score"]
            })
            
    df = pd.DataFrame(dados)
    if not df.empty:
        # Remove duplicatas exatas e ordena
        df = df.drop_duplicates(subset=["Modelo", "Métodos"]).sort_values(by="Acurácia (%)", ascending=False).reset_index(drop=True)
        
        # Exibe o TOP 25 para garantir que as Árvores apareçam
        print("\n🏆 RANKING COMPARATIVO FINAL (TOP 25):")
        print(df.head(25).to_markdown(index=False)) # Removi o index para a tabela ficar mais limpa
        
        df.to_csv(PASTA_RESULTADOS / "TABELA_FINAL_COMPLETA.csv", index=False, sep=";", decimal=",")
        print(f"\n✅ Planilha completa salva em: {PASTA_RESULTADOS}/TABELA_FINAL_COMPLETA.csv")
    else:
        print("❌ Nenhum dado encontrado.")

if __name__ == "__main__":
    compilar_resultados()
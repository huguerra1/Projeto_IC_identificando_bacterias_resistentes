import subprocess
import time
from datetime import datetime

# =============================================================================
# CONFIGURAÇÕES DA BATERIA DE TESTES (CONFORME SUA METODOLOGIA)
# =============================================================================

# Grupo 3: Pré-processamento espectral
preprocessamentos = ['minmax']

# Grupo 4: Extração e seleção de atributos
# Incluímos 'estatisticas' como um método de extração a ser testado
extracoes = ['anova', 'mi', 'estatisticas']

# Scripts do Pipeline
SCRIPT_PROC = '2_processar_e_selecionar.py'
SCRIPT_TREE = '3_treinar_modelo.py'
SCRIPT_XGB  = '3_treinar_xgboost.py'

def log_status(mensagem):
    horario = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{horario}] ➡️ {mensagem}")

# =============================================================================
# LOOP DE EXECUÇÃO
# =============================================================================

start_total = time.time()

log_status("INICIANDO BATERIA DE TESTES COMPLETA (CV=5)")
print("="*75)
print(f"Total de combinações: {len(preprocessamentos) * len(extracoes)}")
print("="*75)

for p in preprocessamentos:
    for e in extracoes:
        start_cenario = time.time()
        print(f"\n🧪 TESTANDO: Pré-proc: {p.upper()} | Extração/Seleção: {e.upper()}")
        print("-" * 45)
        
        try:
            # PASSO 1: Processamento e Seleção
            # Este script agora deve aceitar 'remocao_bandas' e 'estatisticas'
            log_status(f"1/3: Preparando dados ({p} + {e})...")
            subprocess.run(['python', SCRIPT_PROC, '--preproc', p, '--feature', e], check=True)
            
            # PASSO 2: Treinamento Árvore de Decisão (Baseline)
            log_status(f"2/3: Treinando Árvore de Decisão (CV=5)...")
            subprocess.run(['python', SCRIPT_TREE, '--preproc', p, '--feature', e], check=True)
            
            # PASSO 3: Treinamento XGBoost (Alta Performance)
            log_status(f"3/3: Treinando XGBoost (CV=5)...")
            subprocess.run(['python', SCRIPT_XGB, '--preproc', p, '--feature', e], check=True)

            tempo_cenario = (time.time() - start_cenario) / 60
            print(f"✅ Cenário concluído em {tempo_cenario:.2f} min.")

        except subprocess.CalledProcessError as err:
            print(f"❌ ERRO no cenário {p}+{e}: O processo retornou um erro.")
            continue 

print("\n" + "="*75)
tempo_total = (time.time() - start_total) / 60
print(f"🏆 TODOS OS EXPERIMENTOS CONCLUÍDOS!")
print(f"⏱️ Tempo total: {tempo_total:.2f} minutos")
print("="*75)
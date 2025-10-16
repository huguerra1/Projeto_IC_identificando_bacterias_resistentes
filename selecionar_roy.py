import numpy as np
import matplotlib.pyplot as plt
import spectral as sp
from pathlib import Path
import sys

# =============================================================================
# CONFIGURAÇÃO (🚨 EDITE O CAMINHO PARA A IMAGEM QUE VOCÊ QUER ANALISAR)
# =============================================================================
# Coloque aqui o caminho para o arquivo .raw ou .hdr da AMOSTRA BRUTA
# que você quer usar para selecionar a ROI.
CAMINHO_IMAGEM_RAW = 'C:/Users/hugo/OneDrive/identificacao_bacteria/amostras_teste/351_1_240506-160920/capture/351_1_240506-160920.raw'

# =============================================================================

# Variáveis globais para armazenar os cliques
clicks = []

def onclick(event):
    """Função para capturar os eventos de clique do mouse."""
    global clicks
    
    # Ignora cliques fora da área do gráfico
    if event.xdata is None or event.ydata is None:
        return
        
    # Armazena as coordenadas do clique (x, y)
    ix, iy = int(event.xdata), int(event.ydata)
    clicks.append((ix, iy))
    
    print(f'Clique {len(clicks)} registrado em (x={ix}, y={iy})')
    
    # Desenha um ponto no local do clique
    ax.plot(ix, iy, 'r+', markersize=10)
    fig.canvas.draw()
    
    # Se dois cliques foram feitos, calcula e finaliza
    if len(clicks) == 2:
        processar_roi()

def processar_roi():
    """Calcula o centro e o raio e desenha o círculo."""
    global clicks
    
    # O primeiro clique é o centro
    centro_x, centro_y = clicks[0]
    
    # O segundo clique define a borda
    borda_x, borda_y = clicks[1]
    
    # Calcula o raio usando a fórmula da distância
    raio = np.sqrt((borda_x - centro_x)**2 + (borda_y - centro_y)**2)
    raio_int = int(round(raio))
    
    # Desenha o círculo para verificação visual
    circulo = plt.Circle((centro_x, centro_y), raio_int, color='red', fill=False, linewidth=2)
    ax.add_artist(circulo)
    fig.canvas.draw()
    
    print("\n" + "="*50)
    print("✅ ROI Selecionada! Copie o código abaixo e cole no seu script principal:")
    print("="*50)
    print('"roi_coords": {')
    print(f'    "centro": ({centro_y}, {centro_x}),  # (y, x)')
    print(f'    "raio": {raio_int}')
    print('}')
    print("="*50)
    print("\nPode fechar a janela da imagem.")
    
    # Desconecta o evento para não capturar mais cliques
    fig.canvas.mpl_disconnect(cid)

# --- Execução Principal ---
if __name__ == "__main__":
    caminho_raw = Path(CAMINHO_IMAGEM_RAW)
    if not caminho_raw.exists():
        print(f"ERRO: Arquivo não encontrado em '{caminho_raw}'")
        sys.exit(1)
        
    caminho_hdr = caminho_raw.with_suffix('.hdr')
    if not caminho_hdr.exists():
        print(f"ERRO: Arquivo .hdr correspondente não encontrado para '{caminho_raw}'")
        sys.exit(1)

    # Carrega a imagem e seleciona uma banda do meio para visualização
    try:
        img = sp.envi.open(str(caminho_hdr), str(caminho_raw))
        banda_meio = img.nbands // 2
        imagem_banda = img.read_band(banda_meio)
    except Exception as e:
        print(f"ERRO ao carregar a imagem: {e}")
        sys.exit(1)

    # Cria a figura e conecta o evento de clique
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(imagem_banda, cmap='gray')
    ax.set_title('Seleção de ROI: 1º Clique no CENTRO, 2º Clique na BORDA', fontsize=12)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    print("Aguardando cliques na janela da imagem...")
    plt.show()
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import cv2

# Importa os módulos centralizados de configuração e utilidades
import config
import utils

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
def main():
    """
    Função principal que executa a Etapa 1 do pipeline:
    Leitura dos cubos HSI, detecção da ROI, extração dos pixels brutos
    e salvamento em formato .npy.
    """
    # Garante que todos os diretórios de saída necessários existam
    utils.criar_diretorios_necessarios()

    print("="*70)
    print("INICIANDO ETAPA 1: Leitura e Extração de Dados Brutos da ROI")
    print("="*70)

    # Encontra todas as pastas de amostras no diretório de dados originais
    pastas_para_processar = [p for p in config.HSI_ORIGINAL_PATH.iterdir() if p.is_dir()]
    if not pastas_para_processar:
        print(f"❌ Nenhuma pasta de amostra encontrada em: '{config.HSI_ORIGINAL_PATH}'")
        return

    print(f"Encontradas {len(pastas_para_processar)} pastas de amostras para processar.")
    
    # Loop principal para processar cada amostra
    for pasta_amostra in tqdm(pastas_para_processar, desc="Processando Amostras"):
        nome_amostra_base = pasta_amostra.name
        
        try:
            print(f"\nCarregando dados da amostra: '{nome_amostra_base}'...")
            
            pasta_capture = pasta_amostra / "capture"
            arquivos = {
                "raw": f"{nome_amostra_base}.raw",
                "dark": f"DARKREF_{nome_amostra_base}.raw",
                "white": f"WHITEREF_{nome_amostra_base}.raw"
            }
            
            # --- 1. Carregamento e Calibração ---
            # Usa as funções do módulo 'utils'
            raw_cube = utils.carregar_cubo_envi(pasta_capture, arquivos["raw"])
            dark_cube = utils.carregar_cubo_envi(pasta_capture, arquivos["dark"])
            white_cube = utils.carregar_cubo_envi(pasta_capture, arquivos["white"])
            reflectance_cube = utils.calibrar_para_refletancia(raw_cube, dark_cube, white_cube)
            
            # --- 2. Detecção Automática da ROI (Batoque) ---
            print("   Detectando região de interesse (ROI)...")
            mask_2d = utils.encontrar_batoque_hough(reflectance_cube, nome_amostra_base)
            
            # --- 3. Extração e Reorganização dos Dados ---
            # Extrai os pixels que pertencem à máscara da ROI
            matriz_pixels_roi = reflectance_cube[mask_2d == 1]
            
            # O formato agora é (pixels, bandas), que é o padrão para o scikit-learn
            print(f"   Matriz de pixels da ROI extraída com formato (pixels, bandas): {matriz_pixels_roi.shape}")

            # --- 4. Salvamento da Matriz Bruta ---
            nome_arquivo_saida = f'bruta_{nome_amostra_base}.npy'
            caminho_arquivo_saida = config.RAW_MATRICES_PATH / nome_arquivo_saida
            np.save(caminho_arquivo_saida, matriz_pixels_roi)
            
            print(f"✅ Matriz bruta salva com sucesso em: {caminho_arquivo_saida}")

        except (FileNotFoundError, ValueError, TypeError) as e:
            print(f"\n❌ ERRO ao processar '{nome_amostra_base}': {e}. Pulando esta amostra.")
    
    print("\n" + "="*70)
    print("✅ Etapa 1 concluída!")
    print(f"Todas as matrizes brutas foram salvas em: '{config.RAW_MATRICES_PATH}'")
    print("="*70)

    # --- Etapa Final: Visualização de todas as ROIs detectadas ---
    gerar_visualizacao_geral_roi()


def gerar_visualizacao_geral_roi():
    """
    Cria e exibe um plot com todas as imagens de ROI detectadas que foram salvas
    durante o processo.
    """
    visuals_paths = sorted(list(config.ROI_VISUALS_PATH.glob("DETECTADO_*.png")))
    
    if not visuals_paths:
        print("\nNenhuma visualização de ROI foi salva para plotar.")
        return

    print(f"\nGerando plot com as {len(visuals_paths)} imagens de ROI detectadas...")
    
    # Define um grid para as imagens (4 colunas de largura)
    ncols = 4
    nrows = math.ceil(len(visuals_paths) / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3.5))
    fig.suptitle("Visualização de Todos os Batoques Detectados", fontsize=16, y=1.0)
    
    # Garante que 'axes' seja sempre uma lista iterável
    if len(visuals_paths) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Itera sobre os caminhos e os eixos do plot para exibir cada imagem
    for i, img_path in enumerate(visuals_paths):
        ax = axes[i]
        try:
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img_rgb)
            titulo = img_path.stem.replace("DETECTADO_", "")
            ax.set_title(titulo, fontsize=10)
        except Exception as e:
            ax.set_title(f"Falha ao carregar\n{img_path.name}", fontsize=10, color='red')
        
        ax.axis('off')

    # Oculta os eixos extras que não foram usados
    for j in range(len(visuals_paths), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Salva a figura final nos resultados
    figura_final_path = config.RESULTS_PATH / "visualizacao_geral_rois.png"
    plt.savefig(figura_final_path, dpi=150)
    print(f"\n✅ Plot geral das ROIs salvo em: {figura_final_path}")
    plt.show()


if __name__ == "__main__":
    main()
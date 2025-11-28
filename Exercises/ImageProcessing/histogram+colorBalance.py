import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configurações ---
OUTPUT_DIR = "Images/resultados_relatorio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_color_balance(img_rgb, r_factor, g_factor, b_factor):
    """
    Aplica balanço de cor simples (multiplicação de canais).
    IMPORTANTE: Assume que img_rgb já está no formato RGB.
    """
    img_float = img_rgb.astype(np.float32)
    
    # Como convertemos para RGB antes, os índices agora batem com os nomes:
    img_float[:,:,0] = np.clip(img_float[:,:,0] * r_factor, 0, 255) # Canal 0 é Red
    img_float[:,:,1] = np.clip(img_float[:,:,1] * g_factor, 0, 255) # Canal 1 é Green
    img_float[:,:,2] = np.clip(img_float[:,:,2] * b_factor, 0, 255) # Canal 2 é Blue
    
    return img_float.astype(np.uint8)

def equalize_histogram_manual(gray_img):
    """Equalização de histograma implementada manualmente"""
    # Garante que a imagem é 2D (grayscale)
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    hist = np.bincount(gray_img.flatten(), minlength=256)
    num_pixels = gray_img.size
    
    # Calcular CDF (Função de Distribuição Acumulada)
    cdf = hist.cumsum()
    cdf_normalized = cdf / num_pixels
    
    # Mapeamento
    map_values = (cdf_normalized * 255).astype(np.uint8)
    
    # Retorna a imagem equalizada e o histograma original para plotagem
    return map_values[gray_img], hist

# --- Funções de Plotagem ---

def plot_color_comparison(original, warm, cool, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Matplotlib recebe RGB, então as cores ficarão corretas agora
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(warm)
    axes[1].set_title("Balanço Quente")
    axes[2].imshow(cool)
    axes[2].set_title("Balanço Frio")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_equalization(original, equalized, hist_orig, hist_eq, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title("Imagem Original")
    axes[0,1].imshow(equalized, cmap='gray')
    axes[0,1].set_title("Imagem Equalizada")
    
    axes[1,0].bar(range(256), hist_orig, color='black', alpha=0.7)
    axes[1,0].set_title("Histograma Original")
    axes[1,1].bar(range(256), hist_eq, color='black', alpha=0.7)
    axes[1,1].set_title("Histograma Equalizado")
    
    for ax in axes[0,:]: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# --- Main ---

if __name__ == "__main__":
    # 1. Carregar imagem COLORIDA
    img_bgr = cv2.imread('Images/a.jpg')
    
    if img_bgr is None:
        print("Erro: Imagem 'Images/a.jpg' não encontrada.")
    else:
        # CORREÇÃO PRINCIPAL: Converter BGR -> RGB logo após carregar
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        print("Gerando Fig 1: Transformações de Cor...")
        # Agora passamos img_rgb, então a lógica interna (r_factor no canal 0) funcionará
        img_warm = apply_color_balance(img_rgb, 1.4, 1.1, 0.8) # Aumentei um pouco o red para ficar bem visível
        img_cool = apply_color_balance(img_rgb, 0.7, 0.9, 1.3) 
        plot_color_comparison(img_rgb, img_warm, img_cool, "fig_color_balance.png")

    # 2. Carregar imagem GRAYSCALE
    # É importante usar a flag cv2.IMREAD_GRAYSCALE para garantir que carregue com 1 canal apenas
    img_gray = cv2.imread('Images/cat.jpg', cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
         print("Erro: Imagem 'Images/gray.jpg' não encontrada.")
    else:
        print("Gerando Fig 2: Equalização de Histograma...")
        img_eq, hist_orig = equalize_histogram_manual(img_gray)
        # Recalcular hist da equalizada para plot
        hist_eq = np.bincount(img_eq.flatten(), minlength=256)
        plot_equalization(img_gray, img_eq, hist_orig, hist_eq, "fig_histogram_eq.png")
    
    print(f"Resultados salvos em '{OUTPUT_DIR}'")
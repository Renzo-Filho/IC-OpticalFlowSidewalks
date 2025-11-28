import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

def select_image():
    """Seleciona uma imagem através de diálogo de arquivo"""
    root = Tk()
    root.withdraw()  # Esconde a janela principal
    file_path = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    root.destroy()
    return file_path

def simple_color_balance(img, red_factor, green_factor, blue_factor):
    """Método simples de color balance por multiplicação de canais"""
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    r = np.clip(r * red_factor, 0, 255)
    g = np.clip(g * green_factor, 0, 255)
    b = np.clip(b * blue_factor, 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)

def advanced_color_balance(img, temperature=0.0, tint=0.0):
    """
    Método avançado de color balance simulando ajuste de temperatura/tom
    temperature: -1.0 (frio/azul) to 1.0 (quente/amarelo)
    tint: -1.0 (verde) to 1.0 (magenta)
    """
    img_float = img.astype(np.float32) / 255.0
    
    # Conversão simplificada para espaço perceptual (não é XYZ real, mas similar em conceito)
    # Matriz de conversão aproximada RGB para espaço perceptual
    M_to_perceptual = np.array([
        [0.4, 0.3, 0.2],
        [0.2, 0.6, 0.1],
        [0.1, 0.1, 0.7]
    ])
    
    M_from_perceptual = np.linalg.inv(M_to_perceptual)
    
    # Converter para espaço perceptual
    perceptual = np.dot(img_float.reshape(-1, 3), M_to_perceptual.T)
    
    # Ajustar temperatura (canais azul/amarelo) e tom (verde/magenta)
    temp_adjust = np.array([1.0 + temperature * 0.3, 
                          1.0, 
                          1.0 - temperature * 0.3])
    
    tint_adjust = np.array([1.0 + tint * 0.2, 
                          1.0 - tint * 0.3, 
                          1.0 + tint * 0.1])
    
    # Aplicar ajustes
    perceptual_balanced = perceptual * temp_adjust * tint_adjust
    perceptual_balanced = np.clip(perceptual_balanced, 0, 1)
    
    # Converter de volta para RGB
    rgb_balanced = np.dot(perceptual_balanced, M_from_perceptual.T)
    rgb_balanced = np.clip(rgb_balanced, 0, 1)
    
    return (rgb_balanced.reshape(img.shape) * 255).astype(np.uint8)

def analyze_colors(img, title):
    """Analisa e mostra histogramas de cores"""
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(12, 4))
    
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color, label=f'Canal {color.upper()}')
    
    plt.title(f'Histograma - {title}')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Selecionar imagem
    image_path = select_image()
    if not image_path:
        print("Nenhuma imagem selecionada.")
        return
    
    print(f"Carregando: {os.path.basename(image_path)}")
    
    # Carregar imagem
    original = cv2.imread(image_path)
    if original is None:
        print("Erro ao carregar a imagem!")
        return
    
    # Redimensionar se muito grande (para melhor visualização)
    height, width = original.shape[:2]
    if max(height, width) > 1200:
        scale = 1200 / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        original = cv2.resize(original, new_size)
    
    # Aplicar métodos de color balance
    print("\nAplicando métodos de color balance...")
    
    # Método simples - ajuste "quente"
    simple_warm = simple_color_balance(original, 1.3, 1.1, 0.8)
    
    # Método avançado - ajuste de temperatura
    advanced_warm = advanced_color_balance(original, temperature=0.7, tint=0.1)
    
    # Método simples - ajuste "frio"
    simple_cool = simple_color_balance(original, 0.8, 0.9, 1.2)
    
    # Método avançado - ajuste frio
    advanced_cool = advanced_color_balance(original, temperature=-0.6, tint=-0.1)
    
    # Mostrar resultados
    plt.figure(figsize=(20, 12))
    
    # Converter BGR para RGB para matplotlib
    images = [
        ("Original", cv2.cvtColor(original, cv2.COLOR_BGR2RGB)),
        ("Simples - Quente", cv2.cvtColor(simple_warm, cv2.COLOR_BGR2RGB)),
        ("Avançado - Quente", cv2.cvtColor(advanced_warm, cv2.COLOR_BGR2RGB)),
        ("Simples - Frio", cv2.cvtColor(simple_cool, cv2.COLOR_BGR2RGB)),
        ("Avançado - Frio", cv2.cvtColor(advanced_cool, cv2.COLOR_BGR2RGB))
    ]
    
    for i, (title, img_rgb) in enumerate(images, 1):
        plt.subplot(2, 3, i)
        plt.imshow(img_rgb)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Analisar histogramas
    print("\nAnalisando histogramas...")
    analyze_colors(original, "Original")
    analyze_colors(simple_warm, "Simples - Quente")
    analyze_colors(advanced_warm, "Avançado - Quente")
    
    # Salvar resultados se desejar
    save = input("\nDeseja salvar as imagens processadas? (s/n): ").lower()
    if save == 's':
        output_dir = "color_balance_results"
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(f"{output_dir}/original.jpg", original)
        cv2.imwrite(f"{output_dir}/simple_warm.jpg", simple_warm)
        cv2.imwrite(f"{output_dir}/advanced_warm.jpg", advanced_warm)
        cv2.imwrite(f"{output_dir}/simple_cool.jpg", simple_cool)
        cv2.imwrite(f"{output_dir}/advanced_cool.jpg", advanced_cool)
        
        print(f"Imagens salvas em: {output_dir}/")

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    """
    Calcula o histograma de uma imagem (colorida ou grayscale)
    
    Parâmetros:
    - image: imagem (grayscale ou colorida BGR)
    
    Retorna:
    - histograma(s) dependendo do tipo de imagem
    """
    if len(image.shape) == 3:
        # Imagem colorida - calcular histograma para cada canal
        histograms = []
        for channel in range(3):
            hist = np.zeros(256, dtype=int)
            channel_data = image[:, :, channel]
            height, width = channel_data.shape
            for y in range(height):
                for x in range(width):
                    hist[channel_data[y, x]] += 1
            histograms.append(hist)
        return histograms
    else:
        # Imagem grayscale
        hist = np.zeros(256, dtype=int)
        height, width = image.shape
        for y in range(height):
            for x in range(width):
                hist[image[y, x]] += 1
        return hist

def equalize_histogram(image, method='rgb'):
    """
    Aplica equalização de histograma usando diferentes métodos
    
    Parâmetros:
    - image: imagem de entrada (grayscale ou colorida)
    - method: 'rgb' (canal por canal) ou 'lab' (apenas canal L) - ignorado para grayscale
    
    Retorna:
    - imagem equalizada
    """
    if len(image.shape) == 2:
        # Imagem grayscale - usa o método padrão (method é ignorado)
        print("Imagem em escala de cinza detectada - aplicando equalização padrão")
        return _equalize_channel(image)
    
    if method == 'rgb':
        # Equalização por canal RGB
        return _equalize_rgb(image)
    elif method == 'lab':
        # Equalização no espaço LAB (apenas canal L)
        return _equalize_lab(image)
    else:
        raise ValueError("Método deve ser 'rgb' ou 'lab'")

def _equalize_rgb(image):
    """
    Equaliza imagem colorida canal por canal (RGB)
    """
    equalized_image = np.zeros_like(image)
    for channel in range(3):
        channel_data = image[:, :, channel]
        equalized_channel = _equalize_channel(channel_data)
        equalized_image[:, :, channel] = equalized_channel
    return equalized_image

def _equalize_lab(image):
    """
    Equaliza imagem colorida no espaço LAB (apenas canal L)
    """
    # Converter BGR para LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Separar canais LAB
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Equalizar apenas o canal L (luminância)
    l_equalized = _equalize_channel(l_channel)
    
    # Recombinar canais
    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    
    # Converter de volta para BGR
    equalized_image = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    
    return equalized_image

def _equalize_channel(channel_data):
    """
    Equaliza um único canal usando a fórmula recursiva
    c(I) = c(I-1) + (1/N) * h(I)
    """
    # Calcular histograma do canal
    hist = np.zeros(256, dtype=int)
    height, width = channel_data.shape
    total_pixels = height * width
    
    for y in range(height):
        for x in range(width):
            hist[channel_data[y, x]] += 1
    
    # Calcular função de distribuição cumulativa (CDF)
    cdf = np.zeros(256, dtype=float)
    cdf[0] = hist[0] / total_pixels
    
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + (hist[i] / total_pixels)
    
    # Aplicar a equalização: mapear valores usando CDF
    equalized_channel = np.zeros_like(channel_data)
    for y in range(height):
        for x in range(width):
            original_value = channel_data[y, x]
            equalized_value = int(cdf[original_value] * 255)
            equalized_channel[y, x] = equalized_value
    
    return equalized_channel

def plot_comparison(original, equalized, title="Comparação de Equalização"):
    """
    Plota comparação entre imagem original e equalizada
    """
    is_color = len(original.shape) == 3
    
    if is_color:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plotar imagens coloridas
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Imagem Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Imagem Equalizada')
        axes[0, 1].axis('off')
        
        axes[0, 2].axis('off')  # Espaço vazio para alinhamento
        
        # Plotar histogramas coloridos
        hist_orig = calculate_histogram(original)
        hist_eq = calculate_histogram(equalized)
        
        axes[1, 0].bar(range(256), hist_orig[0], color='blue', alpha=0.5, label='Azul', width=1.0)
        axes[1, 0].bar(range(256), hist_orig[1], color='green', alpha=0.5, label='Verde', width=1.0)
        axes[1, 0].bar(range(256), hist_orig[2], color='red', alpha=0.5, label='Vermelho', width=1.0)
        axes[1, 0].set_title('Histograma Original')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(range(256), hist_eq[0], color='blue', alpha=0.5, label='Azul', width=1.0)
        axes[1, 1].bar(range(256), hist_eq[1], color='green', alpha=0.5, label='Verde', width=1.0)
        axes[1, 1].bar(range(256), hist_eq[2], color='red', alpha=0.5, label='Vermelho', width=1.0)
        axes[1, 1].set_title('Histograma Equalizado')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].axis('off')  # Espaço vazio para alinhamento
        
    else:
        # Imagem grayscale
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plotar imagens grayscale
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Imagem Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(equalized, cmap='gray')
        axes[0, 1].set_title('Imagem Equalizada')
        axes[0, 1].axis('off')
        
        # Plotar histogramas grayscale
        hist_orig = calculate_histogram(original)
        hist_eq = calculate_histogram(equalized)
        
        axes[1, 0].bar(range(256), hist_orig, color='gray', alpha=0.7, width=1.0)
        axes[1, 0].set_title('Histograma Original')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].bar(range(256), hist_eq, color='gray', alpha=0.7, width=1.0)
        axes[1, 1].set_title('Histograma Equalizado')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_comparison_both_methods(original, equalized_rgb, equalized_lab, title="Comparação de Métodos"):
    """
    Plota comparação entre os métodos de equalização (apenas para coloridas)
    """
    if len(original.shape) == 2:
        print("Aviso: Comparação de métodos só está disponível para imagens coloridas")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plotar imagens
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Imagem Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(equalized_rgb, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Equalização RGB (canal por canal)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(equalized_lab, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Equalização LAB (apenas luminância)')
    axes[0, 2].axis('off')
    
    # Plotar histogramas
    hist_orig = calculate_histogram(original)
    hist_rgb = calculate_histogram(equalized_rgb)
    hist_lab = calculate_histogram(equalized_lab)
    
    # Histograma Original
    axes[1, 0].bar(range(256), hist_orig[0], color='blue', alpha=0.5, label='Azul', width=1.0)
    axes[1, 0].bar(range(256), hist_orig[1], color='green', alpha=0.5, label='Verde', width=1.0)
    axes[1, 0].bar(range(256), hist_orig[2], color='red', alpha=0.5, label='Vermelho', width=1.0)
    axes[1, 0].set_title('Histograma Original')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histograma RGB
    axes[1, 1].bar(range(256), hist_rgb[0], color='blue', alpha=0.5, label='Azul', width=1.0)
    axes[1, 1].bar(range(256), hist_rgb[1], color='green', alpha=0.5, label='Verde', width=1.0)
    axes[1, 1].bar(range(256), hist_rgb[2], color='red', alpha=0.5, label='Vermelho', width=1.0)
    axes[1, 1].set_title('Histograma RGB Equalizado')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Histograma LAB
    axes[1, 2].bar(range(256), hist_lab[0], color='blue', alpha=0.5, label='Azul', width=1.0)
    axes[1, 2].bar(range(256), hist_lab[1], color='green', alpha=0.5, label='Verde', width=1.0)
    axes[1, 2].bar(range(256), hist_lab[2], color='red', alpha=0.5, label='Vermelho', width=1.0)
    axes[1, 2].set_title('Histograma LAB Equalizado')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
if __name__ == "__main__":
    # Carregar imagem (pode ser colorida ou grayscale)
    image_path = "Images/gray.jpg"  # Altere para o caminho da sua imagem
    
    # Tentar carregar como colorida
    image = cv2.imread(image_path)
    
    if image is None:
        # Tentar carregar como grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Erro: Não foi possível carregar a imagem")
    else:
        print(f"Imagem carregada: {image.shape}")
        print(f"Tipo: {'Colorida' if len(image.shape) == 3 else 'Grayscale'}")
        
        if len(image.shape) == 3:
            # Imagem colorida - aplicar ambos os métodos
            equalized_rgb = equalize_histogram(image, method='rgb')
            equalized_lab = equalize_histogram(image, method='lab')
            
            # Mostrar comparação completa
            plot_comparison_both_methods(image, equalized_rgb, equalized_lab, "Comparação de Métodos")
            
            # Mostrar comparação individual de cada método
            plot_comparison(image, equalized_rgb, "Equalização RGB")
            plot_comparison(image, equalized_lab, "Equalização LAB")
            
            # Salvar imagens equalizadas
            cv2.imwrite("equalizada_rgb.jpg", equalized_rgb)
            cv2.imwrite("equalizada_lab.jpg", equalized_lab)
            
        else:
            # Imagem grayscale - aplicar equalização padrão
            equalized = equalize_histogram(image)
            
            # Mostrar comparação
            plot_comparison(image, equalized, "Equalização de Imagem em Escala de Cinza")
            
            # Salvar imagem equalizada
            cv2.imwrite("equalizada_cinza.jpg", equalized)
        
        print("Processamento concluído!")
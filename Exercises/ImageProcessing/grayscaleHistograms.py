import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_grayscale_image(width, height, pattern_type='gradient', intensity=128):
    """
    Gera uma imagem em escala de cinza com diferentes padrões
    
    Parâmetros:
    - width: largura da imagem
    - height: altura da imagem
    - pattern_type: tipo de padrão ('gradient', 'solid', 'checkerboard', 'random')
    - intensity: intensidade para padrão sólido (0-255)
    
    Retorna:
    - imagem em escala de cinza (numpy array)
    """
    
    # Criar array vazio para a imagem
    image = np.zeros((height, width), dtype=np.uint8)
    
    if pattern_type == 'gradient':
        # Gradiente horizontal de preto para branco
        for x in range(width):
            intensity_value = int((x / width) * 255)
            image[:, x] = intensity_value
            
    elif pattern_type == 'solid':
        # Imagem sólida com intensidade constante
        image[:, :] = intensity
        
    elif pattern_type == 'checkerboard':
        # Tabuleiro de xadrez
        square_size = max(width, height) // 8
        for y in range(height):
            for x in range(width):
                square_x = x // square_size
                square_y = y // square_size
                if (square_x + square_y) % 2 == 0:
                    image[y, x] = 0  # Preto
                else:
                    image[y, x] = 255  # Branco
                    
    elif pattern_type == 'random':
        # Ruído aleatório
        image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        
    elif pattern_type == 'vertical_bars':
        # Barras verticais
        bar_width = width // 4
        intensities = [0, 85, 170, 255]
        for i in range(4):
            start_x = i * bar_width
            end_x = min((i + 1) * bar_width, width)
            image[:, start_x:end_x] = intensities[i]
            
    elif pattern_type == 'circle':
        # Círculo no centro
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= radius:
                    image[y, x] = 255  # Círculo branco
                else:
                    image[y, x] = 0    # Fundo preto
    
    else:
        raise ValueError("Tipo de padrão não reconhecido. Use: 'gradient', 'solid', 'checkerboard', 'random', 'vertical_bars', 'circle'")
    
    return image

def calculate_histogram(image):
    """Calcula o histograma de uma imagem em escala de cinza"""
    hist = np.zeros(256, dtype=int)
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            intensity = image[y, x]
            hist[intensity] += 1
    return hist

def plotHistogram(histogram, image_name):
    """Plota o histograma da imagem"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(range(256), histogram, color='gray', alpha=0.7)
    ax.set_title(f'Histograma - {image_name}')
    ax.set_xlabel('Intensidade')
    ax.set_ylabel('Frequência')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Gerar diferentes tipos de imagens
    patterns = ['gradient', 'solid', 'checkerboard', 'random', 'vertical_bars', 'circle']
    
    for pattern in patterns:
        # Gerar imagem
        img = generate_grayscale_image(400, 300, pattern)
        
        # Calcular histograma
        hist = calculate_histogram(img)
        
        # Mostrar imagem e histograma
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Imagem: {pattern}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(256), hist, color='gray', alpha=0.7)
        plt.title('Histograma')
        plt.xlabel('Intensidade')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Salvar imagem (opcional)
        cv2.imwrite(f"Images/grayscale_{pattern}.png", img)
        print(f"Imagem '{pattern}' salva como 'grayscale_{pattern}.png'")
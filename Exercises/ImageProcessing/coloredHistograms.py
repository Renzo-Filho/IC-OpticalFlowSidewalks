import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_color_histogram(image):
    """
    Calcula o histograma para uma imagem colorida (BGR)
    
    Parâmetros:
    - image: imagem colorida em formato BGR (OpenCV)
    
    Retorna:
    - Tuple com três histogramas (blue_hist, green_hist, red_hist)
    """
    # Separar os canais de cor
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    
    # Calcular histogramas para cada canal
    blue_hist = np.zeros(256, dtype=int)
    green_hist = np.zeros(256, dtype=int)
    red_hist = np.zeros(256, dtype=int)
    
    height, width = image.shape[:2]
    
    for y in range(height):
        for x in range(width):
            blue_hist[blue_channel[y, x]] += 1
            green_hist[green_channel[y, x]] += 1
            red_hist[red_channel[y, x]] += 1
            
    return blue_hist, green_hist, red_hist

def plot_image_with_histogram(image, blue_hist, green_hist, red_hist, image_name=""):
    """
    Plota a imagem ao lado do histograma colorido
    
    Parâmetros:
    - image: imagem colorida em formato BGR
    - blue_hist: histograma do canal azul
    - green_hist: histograma do canal verde  
    - red_hist: histograma do canal vermelho
    - image_name: nome da imagem para o título
    """
    # Criar figura com 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plotar a imagem (convertendo BGR para RGB para matplotlib)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Imagem')
    ax1.axis('off')
    
    # Plotar histograma colorido
    ax2.bar(range(256), blue_hist, color='blue', alpha=0.5, label='Azul', width=1.0)
    ax2.bar(range(256), green_hist, color='green', alpha=0.5, label='Verde', width=1.0)
    ax2.bar(range(256), red_hist, color='red', alpha=0.5, label='Vermelho', width=1.0)
    
    ax2.set_title('Histograma da Imagem')
    ax2.set_xlabel('Intensidade')
    ax2.set_ylabel('Frequência')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
if __name__ == "__main__":
    # Carregar sua imagem colorida
    image_path = "Images/a.jpg"  # Altere para o caminho da sua imagem
    image = cv2.imread(image_path)
    
    if image is not None:
        # Calcular histogramas
        blue_hist, green_hist, red_hist = calculate_color_histogram(image)
        
        # Plotar imagem e histograma lado a lado
        plot_image_with_histogram(image, blue_hist, green_hist, red_hist, "Sua Imagem")
    else:
        print("Erro: Não foi possível carregar a imagem")
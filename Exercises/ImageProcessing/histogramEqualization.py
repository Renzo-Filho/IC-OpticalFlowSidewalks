import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageGenerator:
    """
    Responsável por criar imagens sintéticas para testes (Extraído de grayscaleHistograms.py).
    """
    @staticmethod
    def generate(width, height, pattern_type='gradient', intensity=128):
        image = np.zeros((height, width), dtype=np.uint8)
        
        if pattern_type == 'gradient':
            for x in range(width):
                image[:, x] = int((x / width) * 255)
        elif pattern_type == 'solid':
            image[:, :] = intensity
        elif pattern_type == 'checkerboard':
            square_size = max(width, height) // 8
            for y in range(height):
                for x in range(width):
                    if ((x // square_size) + (y // square_size)) % 2 != 0:
                        image[y, x] = 255
        elif pattern_type == 'random':
            image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        elif pattern_type == 'circle':
            cx, cy = width // 2, height // 2
            radius = min(width, height) // 3
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - cx)**2 + (Y-cy)**2)
            mask = dist_from_center <= radius
            image[mask] = 255
        else:
            raise ValueError("Padrão desconhecido. Use: 'gradient', 'solid', 'checkerboard', 'random', 'circle'")
            
        return image

class HistogramTools:
    """
    Ferramentas de cálculo e manipulação de histogramas (Extraído de cdf.py e coloredHistograms.py).
    """
    @staticmethod
    def calculate(image):
        """Calcula histograma para Grayscale (retorna array) ou Colorido (retorna lista de arrays)."""
        if len(image.shape) == 3: # Colorida
            histograms = []
            for channel in range(3):
                hist = np.zeros(256, dtype=int)
                # Flattening para otimizar o loop manual original
                pixels = image[:, :, channel].flatten()
                for p in pixels:
                    hist[p] += 1
                histograms.append(hist)
            return histograms
        else: # Grayscale
            hist = np.zeros(256, dtype=int)
            pixels = image.flatten()
            for p in pixels:
                hist[p] += 1
            return hist

    @staticmethod
    def _equalize_channel(channel_data):
        """Aplica a equalização baseada na CDF em um único canal."""
        hist = np.bincount(channel_data.flatten(), minlength=256)
        total_pixels = channel_data.size
        
        # Calcular CDF
        cdf = hist.cumsum().astype(float)
        cdf /= total_pixels # Normalizar
        
        # Mapeamento
        lookup_table = (cdf * 255).astype(np.uint8)
        return lookup_table[channel_data]

    @staticmethod
    def equalize(image, method='rgb'):
        """
        Equaliza a imagem.
        Se grayscale: equalização padrão.
        Se colorida: suporta método 'rgb' (canal isolado) ou 'lab' (luminância).
        """
        if len(image.shape) == 2:
            return HistogramTools._equalize_channel(image)
        
        if method == 'rgb':
            eq_img = np.zeros_like(image)
            for i in range(3):
                eq_img[:, :, i] = HistogramTools._equalize_channel(image[:, :, i])
            return eq_img
            
        elif method == 'lab':
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = HistogramTools._equalize_channel(l)
            merged = cv2.merge([l_eq, a, b])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

class Visualizer:
    """
    Centraliza a plotagem de gráficos e comparações.
    """
    @staticmethod
    def plot_simple_analysis(image, title="Análise de Imagem"):
        """Plota imagem e histograma lado a lado (estilo grayscale/coloredHistograms)."""
        is_color = len(image.shape) == 3
        hist = HistogramTools.calculate(image)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Imagem
        if is_color:
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title(title)
        axes[0].axis('off')
        
        # Histograma
        if is_color:
            colors = ['blue', 'green', 'red']
            labels = ['Azul', 'Verde', 'Vermelho']
            for i in range(3):
                axes[1].bar(range(256), hist[i], color=colors[i], alpha=0.5, label=labels[i])
            axes[1].legend()
        else:
            axes[1].bar(range(256), hist, color='gray', alpha=0.7)
            
        axes[1].set_title('Histograma')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_methods(original, eq_rgb, eq_lab):
        """Compara os métodos de equalização RGB vs LAB (Específico para imagens coloridas)."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        imgs = [original, eq_rgb, eq_lab]
        titles = ['Original', 'Equalização RGB (Canais independentes)', 'Equalização LAB (Luminância)']
        
        for i in range(3):
            # Imagens
            axes[0, i].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(titles[i])
            axes[0, i].axis('off')
            
            # Histogramas
            hist = HistogramTools.calculate(imgs[i])
            colors = ['b', 'g', 'r']
            for ch in range(3):
                axes[1, i].plot(hist[ch], color=colors[ch], alpha=0.8)
                axes[1, i].fill_between(range(256), hist[ch], color=colors[ch], alpha=0.1)
            axes[1, i].set_title(f'Histograma: {titles[i]}')
            axes[1, i].grid(True, alpha=0.3)
            
        plt.suptitle("Comparação de Métodos de Equalização", fontsize=16)
        plt.tight_layout()
        plt.show()

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
if __name__ == "__main__":
    print("=== 1. Testando Geração de Imagens (Grayscale) ===")
    gen_img = ImageGenerator.generate(400, 300, 'gradient')
    Visualizer.plot_simple_analysis(gen_img, "Gerada: Gradiente")
    
    print("\n=== 2. Tentando carregar imagem real ===")
    # Substitua pelo caminho da sua imagem
    path = "Images/a.jpg" 
    
    # Se não existir, criamos uma imagem colorida sintética para teste
    try:
        real_img = cv2.imread(path)
        if real_img is None: raise FileNotFoundError
        print(f"Imagem carregada com sucesso: {path}")
    except:
        print("Imagem não encontrada. Gerando imagem colorida aleatória para demonstração.")
        real_img = np.zeros((300, 400, 3), dtype=np.uint8)
        real_img[:, :, 0] = ImageGenerator.generate(400, 300, 'gradient') # Blue
        real_img[:, :, 1] = ImageGenerator.generate(400, 300, 'circle')   # Green
        real_img[:, :, 2] = ImageGenerator.generate(400, 300, 'random')   # Red

    print("\n=== 3. Comparando Métodos de Equalização (RGB vs LAB) ===")
    if len(real_img.shape) == 3:
        eq_rgb = HistogramTools.equalize(real_img, method='rgb')
        eq_lab = HistogramTools.equalize(real_img, method='lab')
        
        Visualizer.compare_methods(real_img, eq_rgb, eq_lab)
        
        # Salvar resultados
        cv2.imwrite("resultado_rgb.jpg", eq_rgb)
        cv2.imwrite("resultado_lab.jpg", eq_lab)
        print("Imagens salvas como 'resultado_rgb.jpg' e 'resultado_lab.jpg'")
    else:
        # Fallback para grayscale
        eq = HistogramTools.equalize(real_img)
        Visualizer.plot_simple_analysis(eq, "Imagem Grayscale Equalizada")
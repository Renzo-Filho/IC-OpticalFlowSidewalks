import cv2
import os
import glob
from pathlib import Path

def criar_video_apartir_imagens(path_imagens, nome_video="video_output.mp4", fps=30):
    """
    Cria um vídeo a partir de imagens em um diretório
    
    Args:
        path_imagens (str): Caminho para o diretório com as imagens
        nome_video (str): Nome do arquivo de vídeo de saída
        fps (int): Frames por segundo do vídeo
    """
    
    # Encontrar todas as imagens no diretório
    extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    lista_imagens = []
    
    for extensao in extensoes:
        lista_imagens.extend(glob.glob(os.path.join(path_imagens, extensao)))
    
    # Ordenar as imagens por nome
    lista_imagens.sort()
    
    if not lista_imagens:
        print("Nenhuma imagem encontrada no diretório especificado.")
        return
    
    print(f"Encontradas {len(lista_imagens)} imagens")
    
    # Ler a primeira imagem para obter as dimensões
    primeira_imagem = cv2.imread(lista_imagens[0])
    if primeira_imagem is None:
        print(f"Erro ao ler a primeira imagem: {lista_imagens[0]}")
        return
    
    altura, largura, _ = primeira_imagem.shape
    
    # Configurar o codec e criar objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    video = cv2.VideoWriter(nome_video, fourcc, fps, (largura, altura))
    
    print(f"Criando vídeo: {nome_video}")
    print(f"Resolução: {largura}x{altura}")
    print(f"FPS: {fps}")
    
    # Processar cada imagem
    for i, caminho_imagem in enumerate(lista_imagens):
        imagem = cv2.imread(caminho_imagem)
        if imagem is None:
            print(f"Erro ao ler imagem: {caminho_imagem}")
            continue
        
        # Redimensionar se necessário (mantém proporção)
        if imagem.shape[1] != largura or imagem.shape[0] != altura:
            imagem = cv2.resize(imagem, (largura, altura))
        
        video.write(imagem)
        
        # Progresso
        if (i + 1) % 10 == 0 or (i + 1) == len(lista_imagens):
            print(f"Processado: {i + 1}/{len(lista_imagens)} imagens")
    
    video.release()
    cv2.destroyAllWindows()
    
    print(f"Vídeo criado com sucesso: {nome_video}")
    print(f"Tamanho do arquivo: {os.path.getsize(nome_video) / (1024*1024):.2f} MB")

def main():
    # Defina o path das imagens aqui
    path_imagens = "./Dataset/cars6-viz"  
    
    # Verificar se o diretório existe
    if not os.path.exists(path_imagens):
        print(f"Diretório não encontrado: {path_imagens}")
        return
    
    nome_video = "Dataset/cars6.mp4"
    fps = 24 
    
    criar_video_apartir_imagens(path_imagens, nome_video, fps)

if __name__ == "__main__":
    main()
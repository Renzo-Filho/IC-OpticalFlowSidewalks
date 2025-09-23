import cv2
import os 
import numpy as np

def process_optical_flow(frame1_path, frame2_path):
    """Processa o fluxo óptico entre dois frames e retorna o frame resultante."""
    # --- Leitura das imagens ---
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    frame1_grayscale = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_grayscale = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # --- Encontra bons pontos para o cálculo ---
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(frame1_grayscale, mask=None, **feature_params)

    # Parâmetros para o Lucas-Kanade
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # --- Cálculo do fluxo óptico ---
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_grayscale, frame2_grayscale, p0, None, **lk_params)

    # Seleciona apenas os "bons pontos"
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    # --- Visualização do fluxo óptico ---
    mask = np.zeros_like(frame1)
    color = np.random.randint(0, 255, (100, 3))

    # Desenha as trilhas
    frame_desenho = frame2.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame_desenho = cv2.circle(frame_desenho, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    # Combina a imagem original com a máscara que contém as linhas
    img_final = cv2.add(frame_desenho, mask)
    
    return img_final

def get_image_paths(directory):
    """Obtém todos os caminhos de imagem de um diretório e os ordena."""
    image_paths = []
    with os.scandir(directory) as images:
        for img in images:
            if img.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(img.path)
    
    image_paths.sort()
    return image_paths

def create_optical_flow_video(image_directory, output_video_path, fps=20):
    """
    Cria um vídeo com fluxo óptico a partir de uma sequência de imagens.
    
    Args:
        image_directory (str): Diretório contendo as imagens
        output_video_path (str): Caminho para salvar o vídeo
        fps (int): Frames por segundo do vídeo
    """
    # Obtém os caminhos das imagens
    image_paths = get_image_paths(image_directory)
    
    if len(image_paths) < 2:
        print("Erro: É necessário pelo menos 2 imagens para criar o fluxo óptico.")
        return
    
    # Lista para armazenar todos os frames processados
    frames = []
    frame_size = None

    print("Processando fluxo óptico...")
    for i in range(len(image_paths)-1):
        print(f"Processando par {i+1}/{len(image_paths)-1}: {os.path.basename(image_paths[i])} -> {os.path.basename(image_paths[i+1])}")
        
        # Processa o fluxo óptico entre os frames consecutivos
        result_frame = process_optical_flow(image_paths[i], image_paths[i+1])
        frames.append(result_frame)
        
        # Define o tamanho do frame a partir do primeiro frame processado
        if frame_size is None:
            frame_size = (result_frame.shape[1], result_frame.shape[0])

    # Cria o vídeo
    print("Criando vídeo...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Escreve todos os frames no vídeo
    for frame in frames:
        out.write(frame)

    # Libera o objeto VideoWriter
    out.release()
    
    print(f"Vídeo criado com sucesso: {output_video_path}")
    print(f"Total de frames processados: {len(frames)}")
    print(f"Tamanho do vídeo: {frame_size[0]}x{frame_size[1]}")
    print(f"FPS: {fps}")

def play_video(video_path):
    """Reproduz um vídeo usando OpenCV."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return
    
    print("Reproduzindo vídeo. Pressione 'q' para sair.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Fluxo Optico - Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Função principal."""
    # Configurações
    image_directory = 'Dataset/videoFrames'
    output_video_path = 'optical_flow_video2.mp4'
    fps = 20
    
    # Cria o vídeo com fluxo óptico
    create_optical_flow_video(image_directory, output_video_path, fps)
    
    # Pergunta se deseja reproduzir o vídeo
    play_video_option = input("Deseja reproduzir o vídeo? (s/n): ").lower()
    if play_video_option == 's':
        play_video(output_video_path)

if __name__ == "__main__":
    main()
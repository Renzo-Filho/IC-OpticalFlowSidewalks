import cv2
import os 
import numpy as np

video_path = 'Dataset/my_video.mp4' 
output_dir = 'Dataset/videoFrames' 

def convertToFrames(video_path, output_dir):
    
    # Passo 1: Verificar e criar o diretório de saída, se necessário.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Diretório criado com sucesso.")

    # Cria um objeto de captura de vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo no caminho: {video_path}")
    else:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret:
                # Constrói o nome do arquivo para o frame atual
                # Ex: frame_0000.png, frame_0001.png, etc.
                # O f-string :04d garante que o número tenha sempre 4 dígitos (0001, 0010, 0100...)
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
                
                # Salva o frame atual como um arquivo de imagem
                cv2.imwrite(frame_filename, frame)
                
                frame_count += 1
            else:
                break

    cap.release()
    cv2.destroyAllWindows() 

    print(f"\nTotal de {frame_count} frames salvos em '{output_dir}'.")

""" 
1. Carregará duas imagens consecutivas do nosso diretório Dataset/videoFrames.

2. Converterá ambas para escala de cinza.

3. Usará a função cv2.goodFeaturesToTrack() no primeiro frame para encontrar uma lista de cantos promissores.

4. Para visualização e verificação, vamos desenhar pequenos círculos sobre os cantos detectados na primeira 
imagem e exibi-la na tela, para que possamos ver se a detecção fez sentido. 
"""

def prepFrames(frame1_path, frame2_path):
    """ Carregará duas imagens consecutivas do nosso diretório Dataset/videoFrames e converterá ambas para escala de cinza. """

    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    frame1_grayscale = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_grayscale = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)



    # Parâmetros para o detector de cantos Shi-Tomasi
    feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

    # A função cv2.goodFeaturesToTrack encontra os cantos.
    # p0 é uma lista de pontos (x, y) detectados.
    p0 = cv2.goodFeaturesToTrack(frame1_grayscale, mask = None, **feature_params)
    print(f"{len(p0)} pontos encontrados.")

    """
    # --- Visualização dos pontos encontrados ---
    # Cria uma cópia da imagem original para desenhar sobre ela
    frame_desenho = frame1.copy()

    # Percorre cada ponto encontrado em p0
    for point in p0:
        # Extrai as coordenadas x e y do ponto
        x, y = point.ravel()
        # Desenha um pequeno círculo azul no local do ponto
        cv2.circle(frame_desenho, (int(x), int(y)), 5, (255, 0, 0), -1)

    # Mostra a imagem com os pontos desenhados
    cv2.imshow('Cantos Detectados - Shi-Tomasi', frame_desenho)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    # Parâmetros para o Lucas-Kanade
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calcula o fluxo óptico
    # p1 são as novas posições dos pontos de p0
    # st é o status: 1 se o ponto foi rastreado com sucesso, 0 caso contrário
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_grayscale, frame2_grayscale, p0, None, **lk_params)

    # Seleciona apenas os "bons pontos" (aqueles que foram rastreados com sucesso)
    # st == 1 cria uma máscara booleana. Usamos para filtrar p1 e p0.
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    # --- Visualização do fluxo óptico ---
    # Cria uma máscara (imagem preta) para desenhar as linhas de rastreamento
    mask = np.zeros_like(frame1)
    
    # Cria uma cor aleatória para as linhas
    color = np.random.randint(0, 255, (100, 3))

    # Desenha as trilhas
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Desenha a linha do ponto antigo para o novo (a trilha do movimento)
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        # Desenha um círculo na nova posição do ponto
        frame_desenho = cv2.circle(frame2, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    # Combina a imagem original com a máscara que contém as linhas
    img_final = cv2.add(frame_desenho, mask)

    cv2.imshow('Fluxo Optico - Lucas-Kanade', img_final)
    print("Pressione qualquer tecla para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()




f1 = "Dataset/videoFrames/frame_0080.png"
f2 = "Dataset/videoFrames/frame_0081.png"

prepFrames(f1, f2)
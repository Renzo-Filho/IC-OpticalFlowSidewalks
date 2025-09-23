import cv2
import os 
import numpy as np

def LKmethod(frame1_path, frame2_path):
    """ Carrega duas imagens consecutivas do nosso diretório Dataset/videoFrames e converterá ambas para escala de cinza. 
    Depois, usamos a função cv2.goodFeaturesToTrack() no primeiro frame para encontrar uma lista de cantos promissores. 
    Por fim, calculamos o fluxo óptico. """

    # --- Leitura das imagens ---

    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    frame1_grayscale = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_grayscale = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # --- Encontra bons pontos para o cálculo ---

    # Parâmetros para o detector de cantos Shi-Tomasi
    feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

    # A função cv2.goodFeaturesToTrack encontra os cantos. p0 é uma lista de pontos (x, y) detectados.
    p0 = cv2.goodFeaturesToTrack(frame1_grayscale, mask = None, **feature_params)
    print(f"{len(p0)} pontos encontrados.")

    # Parâmetros para o Lucas-Kanade
    lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # --- Cálculo do fluxo óptico ---

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

#####################################################################################3

dir = 'Dataset/cars6'
D = []

with os.scandir(dir) as images:
    for img in images:
        D.append(img.path)

D.sort()

for i in range(len(D)-1):
    LKmethod(D[i], D[i+1])

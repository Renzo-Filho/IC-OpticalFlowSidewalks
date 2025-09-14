import numpy as np
import cv2
import os

video_path = 'Dataset/inova.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo: {video_path}")
    exit()

"""
# Parâmetros para detecção de cantos (Shi-Tomasi)
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
"""

feature_params = dict( maxCorners = 100,  
                       qualityLevel = 0.05,   
                       minDistance = 5, 
                       blockSize = 7 )

# Parâmetros para o fluxo óptico (Lucas-Kanade)
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Gera cores aleatórias para visualização
color = np.random.randint(0, 255, (100, 3))

# Pega o primeiro frame e encontra os cantos
ret, old_frame = cap.read()
if not ret:
    print("Não foi possível ler o primeiro frame do vídeo.")
    exit()
    
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# A máscara agora não é mais necessária para acumular trilhas,
# então não a usaremos no loop de desenho.

while True:
    # Lê um novo frame
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo.")
        break
        
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcula o fluxo óptico
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Filtra os pontos que foram rastreados com sucesso
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        continue

    # --- NOVA LÓGICA DE VISUALIZAÇÃO ---
    # Cria uma cópia do frame para desenhar sobre ele
    frame_com_vetores = frame.copy()

    # Desenha os vetores (setas)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        # Desenha a seta do ponto antigo para o novo
        # cv2.arrowedLine(imagem, ponto_inicial, ponto_final, cor, espessura)
        frame_com_vetores = cv2.arrowedLine(frame_com_vetores, (int(c), int(d)), (int(a), int(b)), color[i].tolist(), 2)
        # Desenha um círculo na nova posição do ponto
        frame_com_vetores = cv2.circle(frame_com_vetores, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    # Exibe o resultado com os vetores instantâneos
    cv2.imshow('Campo Vetorial - Fluxo Optico', frame_com_vetores)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # ATUALIZAÇÃO PARA O PRÓXIMO CICLO
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Limpeza final
cv2.destroyAllWindows()
cap.release()
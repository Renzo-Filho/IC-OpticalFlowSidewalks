import numpy as np
import cv2
import os

video_path = 'Dataset/my_video.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo: {video_path}")
    exit()

# Parâmetros para detecção de cantos (Shi-Tomasi)
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
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

# Cria uma máscara para desenhar as trilhas
mask = np.zeros_like(old_frame)


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
    else: # Se nenhum ponto for rastreado, reinicia o loop
        # (Opcional: poderia re-detectar features aqui)
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        continue # Pula o resto do loop e vai para a próxima iteração
        
    # Desenha as trilhas
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    # Exibe o resultado
    img = cv2.add(frame, mask)
    cv2.imshow('Rastreamento com Fluxo Optico', img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # 3. ATUALIZAÇÃO PARA O PRÓXIMO CICLO
    # O frame atual se torna o frame anterior
    old_gray = frame_gray.copy()
    # Os pontos atuais se tornam os pontos anteriores. O reshape é necessário.
    p0 = good_new.reshape(-1, 1, 2)

# Limpeza final
cv2.destroyAllWindows()
cap.release()
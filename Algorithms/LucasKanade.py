import cv2 as cv
import numpy as np
import os

class LucasKanade:
    """
    Classe completa para Rastreamento de Fluxo Óptico (Lucas-Kanade).
    Suporta entrada de arquivos de vídeo (.mp4, .avi) ou pastas com sequências de imagens.
    Permite visualização em tempo real e salvamento do resultado.
    """

    def __init__(self, input_source):
        """
        Inicializa o rastreador.
        
        :param input_source: Caminho para um arquivo de vídeo OU uma pasta contendo imagens.
        """
        self.input_source = input_source
        self.is_video_file = os.path.isfile(input_source)
        self.image_paths = []
        self.current_img_idx = 0
        self.cap = None

        # --- Configuração da Fonte de Entrada ---
        if self.is_video_file:
            # Modo Vídeo
            self.cap = cv.VideoCapture(input_source)
            if not self.cap.isOpened():
                raise ValueError(f"Não foi possível abrir o vídeo: {input_source}")
        elif os.path.isdir(input_source):
            # Modo Sequência de Imagens
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            # Lista e ordena os arquivos para garantir a ordem temporal
            self.image_paths = sorted([
                os.path.join(input_source, f) for f in os.listdir(input_source)
                if f.lower().endswith(valid_exts)
            ])
            if not self.image_paths:
                raise ValueError(f"Nenhuma imagem encontrada na pasta: {input_source}")
        else:
            raise ValueError(f"Entrada inválida (não é arquivo nem pasta): {input_source}")

        # --- Parâmetros Lucas-Kanade ---
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # --- Parâmetros Shi-Tomasi (Detecção de Cantos) ---
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        # --- Variáveis de Estado ---
        self.prev_gray = None
        self.p0 = None

    def _read_next_frame(self):
        """Abstrai a leitura do próximo frame (seja de vídeo ou lista de imagens)."""
        if self.is_video_file:
            return self.cap.read()
        else:
            if self.current_img_idx < len(self.image_paths):
                frame = cv.imread(self.image_paths[self.current_img_idx])
                self.current_img_idx += 1
                if frame is None:
                    return False, None
                return True, frame
            else:
                return False, None

    def _detect_features(self, gray_frame):
        """Detecta novos pontos de interesse."""
        self.p0 = cv.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
        if self.p0 is None:
            self.p0 = np.array([], dtype=np.float32).reshape(0, 1, 2)

    def _process_frame_logic(self, frame):
        """Lógica matemática do Fluxo Óptico."""
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.p0.shape[0] == 0:
            self._detect_features(gray_frame)
            self.prev_gray = gray_frame
            return np.array([]), np.array([])

        p1, st, err = cv.calcOpticalFlowPyrLK(self.prev_gray, gray_frame, self.p0, None, **self.lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
        else:
            good_new = np.array([])
            good_old = np.array([])

        if len(good_new) < (self.feature_params['maxCorners'] * 0.75):
            self._detect_features(gray_frame)
        else:
            self.p0 = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray_frame.copy()
        return good_old, good_new

    def _draw_visuals(self, frame, good_old, good_new):
        """Desenha os vetores no frame."""
        vis_frame = frame.copy()
        color_arrow = (0, 255, 255) # Amarelo
        color_point = (0, 0, 255)   # Vermelho

        if good_old.size > 0:
            for new, old in zip(good_new, good_old):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)

                if np.hypot(a - c, b - d) < 1.0: continue

                cv.arrowedLine(vis_frame, (c, d), (a, b), color_arrow, 2, cv.LINE_AA, 0, 0.35)
                cv.circle(vis_frame, (c, d), 2, color_point, -1, cv.LINE_AA)

        return vis_frame

    def run(self, save_video=False, output_file='output_flow.mp4', display=True):
        """
        Executa o loop principal.
        
        :param save_video: Se True, salva o resultado em um arquivo de vídeo.
        :param output_file: Nome do arquivo de saída (se save_video=True).
        :param display: Se True, mostra a janela com o vídeo processado.
        """
        print(f"Iniciando processamento de: {self.input_source}")
        if save_video:
            print(f"Gravando saída em: {output_file}")
        
        writer = None

        while True:
            ret, frame = self._read_next_frame()
            if not ret:
                print("Fim do processamento.")
                break

            # 1. Calcular
            good_old, good_new = self._process_frame_logic(frame)

            # 2. Desenhar
            final_image = self._draw_visuals(frame, good_old, good_new)

            # 3. Gravar (Opcional)
            if save_video:
                if writer is None:
                    # Inicializa o VideoWriter no primeiro frame para pegar as dimensões corretas
                    h, w = final_image.shape[:2]
                    fourcc = cv.VideoWriter_fourcc(*'mp4v')
                    # FPS fixo em 20.0, mas idealmente deveria vir do vídeo original se possível
                    writer = cv.VideoWriter(output_file, fourcc, 20.0, (w, h))
                
                writer.write(final_image)

            # 4. Mostrar (Opcional)
            if display:
                cv.imshow('Optical Flow (Lucas-Kanade)', final_image)
                if cv.waitKey(30) & 0xFF == ord('q'):
                    print("Interrompido pelo usuário.")
                    break

        # Limpeza
        if self.cap: self.cap.release()
        if writer: writer.release()
        cv.destroyAllWindows()
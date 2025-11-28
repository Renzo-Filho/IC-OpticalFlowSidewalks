import cv2 as cv
import numpy as np
import os

class Farneback:
    """
    Classe para calcular o Fluxo Óptico Denso (Farneback).
    Exibe: Vídeo Original + Fluxo HSV (Lado a Lado) em Resolução Original.
    """

    def __init__(self, input_source):
        self.input_source = input_source
        self.is_video_file = os.path.isfile(input_source)
        self.image_paths = []
        self.current_img_idx = 0
        self.cap = None

        # --- Configuração da Fonte ---
        if self.is_video_file:
            self.cap = cv.VideoCapture(input_source)
            if not self.cap.isOpened():
                raise ValueError(f"Erro ao abrir: {input_source}")
        elif os.path.isdir(input_source):
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            self.image_paths = sorted([os.path.join(input_source, f) for f in os.listdir(input_source) if f.lower().endswith(valid_exts)])
            if not self.image_paths:
                raise ValueError(f"Pasta vazia: {input_source}")
        else:
            raise ValueError(f"Entrada inválida.")

        # --- Parâmetros Farneback (Otimizados para velocidade) ---
        self.fb_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, 
                              poly_n=5, poly_sigma=1.2, flags=0)

        # --- Variáveis de Estado ---
        self.prev_gray = None
        self.hsv = None
        
        # Cria a legenda (roda de cores)
        self.legend_img = self._create_color_wheel(size=60) 

    def _create_color_wheel(self, size=60):
        """
        Gera a roda de cores usando EXATAMENTE a mesma matemática do OpenCV.
        Isso garante que a legenda bata perfeitamente com o vídeo.
        """
        # Cria um grid de coordenadas (x, y) centrado no 0
        x, y = np.meshgrid(np.arange(-size, size), np.arange(-size, size))
        
        # Usa a mesma função do cálculo de fluxo para garantir consistência
        # No OpenCV, y positivo é para baixo, mas cartToPolar lida com isso.
        mag, angle = cv.cartToPolar(x.astype(np.float32), y.astype(np.float32))
        
        # Máscara circular
        mask = mag <= size
        
        wheel_hsv = np.zeros((size*2, size*2, 3), dtype=np.uint8)
        
        # Hue: Mapeia ângulo (radianos) para 0-180
        wheel_hsv[..., 0] = angle * 180 / np.pi / 2
        
        # Saturation: Máxima (255)
        wheel_hsv[..., 1] = 255
        
        # Value: Máximo (255) para ser bem visível
        wheel_hsv[..., 2] = 255
        
        # Aplica a máscara (deixa preto fora do círculo)
        wheel_hsv[~mask] = 0
        
        wheel_bgr = cv.cvtColor(wheel_hsv, cv.COLOR_HSV2BGR)
        
        # Borda branca para acabamento
        cv.circle(wheel_bgr, (size, size), size, (255, 255, 255), 1)
        
        return wheel_bgr

    def _read_next_frame(self):
        if self.is_video_file:
            return self.cap.read()
        else:
            if self.current_img_idx < len(self.image_paths):
                frame = cv.imread(self.image_paths[self.current_img_idx])
                self.current_img_idx += 1
                return (frame is not None), frame
            return False, None

    def _process_and_draw(self, frame):
        # Guarda o tamanho original
        orig_h, orig_w = frame.shape[:2]

        # 1. Redimensionar (Downscale) para processamento rápido
        scale_factor = 0.5 
        small_w = int(orig_w * scale_factor)
        small_h = int(orig_h * scale_factor)
        frame_small = cv.resize(frame, (small_w, small_h), interpolation=cv.INTER_AREA)
        
        gray_frame = cv.cvtColor(frame_small, cv.COLOR_BGR2GRAY)
        
        # Inicialização
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            # Retorna visualização vazia inicial
            return np.hstack((frame, np.zeros_like(frame)))

        # 2. Calcular Fluxo (na imagem pequena)
        flow = cv.calcOpticalFlowFarneback(self.prev_gray, gray_frame, None, **self.fb_params)

        # 3. Converter para Cores HSV
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Cria array HSV pequeno
        hsv_small = np.zeros_like(frame_small)
        hsv_small[..., 1] = 255
        hsv_small[..., 0] = ang * 180 / np.pi / 2
        hsv_small[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        
        bgr_flow_small = cv.cvtColor(hsv_small, cv.COLOR_HSV2BGR)

        # 4. Redimensionar (Upscale) de volta ao tamanho ORIGINAL
        # Usamos INTER_LINEAR ou CUBIC para suavizar os blocos
        bgr_flow_large = cv.resize(bgr_flow_small, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)

        # 5. Sobrepor a Legenda (Agora na imagem GRANDE)
        l_h, l_w = self.legend_img.shape[:2]
        
        # Desenha no canto inferior direito
        if orig_h > l_h and orig_w > l_w:
            y_offset = orig_h - l_h - 20
            x_offset = orig_w - l_w - 20
            
            roi = bgr_flow_large[y_offset:y_offset+l_h, x_offset:x_offset+l_w]
            
            gray_legend = cv.cvtColor(self.legend_img, cv.COLOR_BGR2GRAY)
            ret, mask = cv.threshold(gray_legend, 10, 255, cv.THRESH_BINARY)
            mask_inv = cv.bitwise_not(mask)
            
            img_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
            img_fg = cv.bitwise_and(self.legend_img, self.legend_img, mask=mask)
            
            dst = cv.add(img_bg, img_fg)
            bgr_flow_large[y_offset:y_offset+l_h, x_offset:x_offset+l_w] = dst

        # 6. Juntar lado a lado (Original | Fluxo Grande)
        combined = np.hstack((frame, bgr_flow_large))
        
        self.prev_gray = gray_frame.copy()
        return combined

    def run(self, save_video=False, output_file='output_comparison.mp4', display=True):
        print(f"Iniciando Farneback (HD) em: {self.input_source}")
        writer = None

        while True:
            ret, frame = self._read_next_frame()
            if not ret: break

            final_image = self._process_and_draw(frame)

            if display:
                cv.imshow('Original vs Fluxo Denso', final_image)
                if cv.waitKey(1) & 0xFF == ord('q'): break
            
            if save_video:
                if writer is None:
                    h, w = final_image.shape[:2]
                    fourcc = cv.VideoWriter_fourcc(*'mp4v')
                    # FPS 20.0 (pode ajustar conforme necessário)
                    writer = cv.VideoWriter(output_file, fourcc, 20.0, (w, h))
                writer.write(final_image)

        if self.cap: self.cap.release()
        if writer: writer.release()
        cv.destroyAllWindows()
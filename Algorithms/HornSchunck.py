import cv2 as cv
import numpy as np
import os

class HornSchunck:
    """
    Classe para calcular o Fluxo Óptico Denso usando o método Horn-Schunck.
    Implementação manual iterativa utilizando NumPy.
    
    Características:
    - Método Global (impõe suavidade em toda a imagem).
    - Visualização: Side-by-Side com Roda de Cores HSV.
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

        # --- Parâmetros Horn-Schunck ---
        # Alpha: Regularização de suavidade. 
        # Maior = fluxo mais suave (borrado). Menor = fluxo mais fiel ao gradiente local.
        self.alpha = 20.0 
        self.iterations = 40 # Iterações por frame (quanto mais, mais preciso e mais lento)
        self.epsilon = 0.001 # Critério de parada (opcional)

        # Variáveis de estado
        self.prev_gray = None
        
        # Gera a legenda (roda de cores)
        self.legend_img = self._create_color_wheel(size=60)

    def _create_color_wheel(self, size=60):
        """Gera a legenda de cores (Mesma lógica do Farneback para consistência)."""
        x, y = np.meshgrid(np.arange(-size, size), np.arange(-size, size))
        mag, angle = cv.cartToPolar(x.astype(np.float32), y.astype(np.float32))
        mask = mag <= size
        
        wheel_hsv = np.zeros((size*2, size*2, 3), dtype=np.uint8)
        wheel_hsv[..., 0] = angle * 180 / np.pi / 2
        wheel_hsv[..., 1] = 255
        wheel_hsv[..., 2] = 255
        wheel_hsv[~mask] = 0
        
        wheel_bgr = cv.cvtColor(wheel_hsv, cv.COLOR_HSV2BGR)
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

    def _compute_horn_schunck(self, img1, img2):
        """
        Implementação matemática manual do algoritmo Horn-Schunck.
        """
        # Converte para float32 para precisão matemática (0 a 255)
        I1 = np.float32(img1)
        I2 = np.float32(img2)

        # 1. Calcular Derivadas (Gradientes)
        # Ix e Iy (Gradientes espaciais) e It (Gradiente temporal)
        # Usamos filtro Sobel ou simples diferenças
        grad_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25 # Kernel 2x2 simples usado no paper original
        
        # Pode-se usar Sobel do OpenCV, mas o kernel de média 2x2 é o clássico do HS
        Ix = cv.filter2D(I1, -1, np.array([[-1, 1], [-1, 1]])*0.25) + \
             cv.filter2D(I2, -1, np.array([[-1, 1], [-1, 1]])*0.25)
             
        Iy = cv.filter2D(I1, -1, np.array([[-1, -1], [1, 1]])*0.25) + \
             cv.filter2D(I2, -1, np.array([[-1, -1], [1, 1]])*0.25)
             
        It = cv.filter2D(I1, -1, np.ones((2,2))*0.25) + \
             cv.filter2D(I2, -1, np.ones((2,2))*-0.25)

        # 2. Inicializar velocidades (u, v) com zeros
        u = np.zeros_like(I1)
        v = np.zeros_like(I1)

        # Kernel de média (Laplaciano) para a parte de suavidade
        # O paper original usa máscara: [[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]]
        avg_kernel = np.array([[0, 1/4, 0], 
                               [1/4, 0, 1/4], 
                               [0, 1/4, 0]], dtype=np.float32)

        # 3. Iterações (Solver Jacobi)
        for _ in range(self.iterations):
            # Calcula médias locais (suavidade)
            u_avg = cv.filter2D(u, -1, avg_kernel)
            v_avg = cv.filter2D(v, -1, avg_kernel)
            
            # Fórmula de atualização iterativa do Horn-Schunck
            # P = (Ix * u_avg + Iy * v_avg + It)
            # D = (alpha^2 + Ix^2 + Iy^2)
            # u = u_avg - Ix * (P/D)
            # v = v_avg - Iy * (P/D)
            
            P = (Ix * u_avg + Iy * v_avg + It)
            D = (self.alpha**2 + Ix**2 + Iy**2)
            ratio = P / (D + 1e-6) # + epsilon para evitar div por zero
            
            u = u_avg - Ix * ratio
            v = v_avg - Iy * ratio

        return u, v

    def _process_and_draw(self, frame):
        orig_h, orig_w = frame.shape[:2]

        # Mantemos o downscale para performance
        scale_factor = 0.75 
        small_w = int(orig_w * scale_factor)
        small_h = int(orig_h * scale_factor)
        frame_small = cv.resize(frame, (small_w, small_h), interpolation=cv.INTER_AREA)
        
        gray_frame = cv.cvtColor(frame_small, cv.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            return np.hstack((frame, np.zeros_like(frame)))

        # 1. Computar Horn-Schunck
        u, v = self._compute_horn_schunck(self.prev_gray, gray_frame)

        # --- CORREÇÃO DE DIREÇÃO ---
        # O carro vai para a direita, mas estava verde (esquerda). 
        # Invertemos os vetores aqui.
        u = -u
        v = -v
        # ---------------------------

        # 2. Visualização
        mag, ang = cv.cartToPolar(u, v)
        
        hsv_small = np.zeros_like(frame_small)
        hsv_small[..., 1] = 255
        hsv_small[..., 0] = ang * 180 / np.pi / 2
        
        # --- SEUS PARÂMETROS AJUSTADOS ---
        # Mantivemos sua lógica de Alpha alto, mas adicionamos um 'threshold'
        # para apagar o ruído do asfalto/árvores que sobra.
        sensitivity = 100.0 
        threshold = 5.0  # Pixels com movimento menor que isso ficam pretos
        
        mag_amplified = mag * sensitivity
        
        # Limpeza de ruído (Thresholding)
        mag_amplified[mag_amplified < threshold] = 0
        
        hsv_small[..., 2] = np.clip(mag_amplified, 0, 255)
        
        bgr_flow_small = cv.cvtColor(hsv_small, cv.COLOR_HSV2BGR)

        # 3. Upscale e Legenda (igual anterior)
        bgr_flow_large = cv.resize(bgr_flow_small, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
        
        # ... (Código de colar legenda e juntar imagens permanece igual) ...
        # Copie o resto da função anterior para a legenda aqui
        l_h, l_w = self.legend_img.shape[:2]
        if orig_h > l_h and orig_w > l_w:
            y_off = orig_h - l_h - 20
            x_off = orig_w - l_w - 20
            roi = bgr_flow_large[y_off:y_off+l_h, x_off:x_off+l_w]
            mask = cv.cvtColor(self.legend_img, cv.COLOR_BGR2GRAY)
            _, mask = cv.threshold(mask, 10, 255, cv.THRESH_BINARY)
            img_bg = cv.bitwise_and(roi, roi, mask=cv.bitwise_not(mask))
            img_fg = cv.bitwise_and(self.legend_img, self.legend_img, mask=mask)
            bgr_flow_large[y_off:y_off+l_h, x_off:x_off+l_w] = cv.add(img_bg, img_fg)

        combined = np.hstack((frame, bgr_flow_large))
        
        self.prev_gray = gray_frame.copy()
        return combined
    
    def run(self, save_video=False, output_file='output_hs.mp4', display=True):
        print(f"Iniciando Horn-Schunck (Global) em: {self.input_source}")
        writer = None

        while True:
            ret, frame = self._read_next_frame()
            if not ret: break

            final_image = self._process_and_draw(frame)

            if display:
                cv.imshow('Original vs Horn-Schunck', final_image)
                if cv.waitKey(1) & 0xFF == ord('q'): break
            
            if save_video:
                if writer is None:
                    h, w = final_image.shape[:2]
                    fourcc = cv.VideoWriter_fourcc(*'mp4v')
                    writer = cv.VideoWriter(output_file, fourcc, 20.0, (w, h))
                writer.write(final_image)

        if self.cap: self.cap.release()
        if writer: writer.release()
        cv.destroyAllWindows()
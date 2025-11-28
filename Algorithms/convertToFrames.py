import cv2
import os

def convertToFrames(video_path, output_dir):
    """ Transforma um vídeo em seus frames. """

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

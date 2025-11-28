from LucasKanade import LucasKanade

# --- CASO 1: Usando uma Pasta de Imagens e Salvando o Vídeo ---
print("--- Teste 1: Pasta de Imagens -> Vídeo ---")
folder_path = 'Dataset/videoFrames' # Substitua pelo caminho real

try:
    # Instancia apontando para a pasta
    tracker_folder = LucasKanade(folder_path)
    
    # Roda: Salva o vídeo, mas NÃO exibe janela (processamento rápido)
    tracker_folder.run(save_video=True, output_file='Outputs/resultado_pasta.mp4', display=False)

except Exception as e:
    print(f"Pulei o teste de pasta: {e}")


# --- CASO 2: Usando um Arquivo de Vídeo e Apenas Exibindo ---
print("\n--- Teste 2: Arquivo de Vídeo -> Tela ---")
video_path = 'Dataset/my_video.mp4' # Substitua pelo caminho real

try:
    # Instancia apontando para o arquivo
    tracker_video = LucasKanade(video_path)
    
    # Roda: Exibe na tela, NÃO salva nada
    tracker_video.run(save_video=False, display=True)

except Exception as e:
    print(f"Pulei o teste de vídeo: {e}")
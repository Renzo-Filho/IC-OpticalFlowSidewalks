from LucasKanade import LucasKanade
from Farneback import Farneback
from HornSchunck import HornSchunck

video_path = 'Dataset/my_video.mp4'

# Escolha: 'lk', 'farneback', 'hs'
algoritmo = 'hs' 

if algoritmo == 'lk':
    tracker = LucasKanade(video_path)
elif algoritmo == 'farneback':
    tracker = Farneback(video_path)
elif algoritmo == 'hs':
    print("Aviso: Horn-Schunck é lento em Python. Usando baixa resolução...")
    tracker = HornSchunck(video_path)

tracker.run(display=True)
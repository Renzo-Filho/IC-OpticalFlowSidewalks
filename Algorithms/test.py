from LucasKanade import LucasKanade
from Farneback import Farneback
from HornSchunck import HornSchunck

video_path = 'Dataset/cars6'
#video_path = 'Dataset/my_video.mp4'

# Escolha: 'lk', 'farneback', 'hs'
algoritmo = 'farneback' 

if algoritmo == 'lk':
    tracker = LucasKanade(video_path)
elif algoritmo == 'farneback':
    tracker = Farneback(video_path)
elif algoritmo == 'hs':
    tracker = HornSchunck(video_path)

tracker.run(save_video=True, output_file='Outputs/farneback_video_cars6.mp4', display=False)
#tracker.run(display=True)
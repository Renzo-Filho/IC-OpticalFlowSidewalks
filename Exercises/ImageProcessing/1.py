import cv2, random
import numpy as np


img = cv2.imread('Images/a.jpg')
height, width, canais = img.shape

print(f"Dimensões: {height}x{width}")
print(f"Número de canais: {canais}")

def showIMG(img):
    cv2.imshow('Titulo here', img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  


b, g, r = cv2.split(img)

# Acessar pixel específico (linha, coluna)
valor_azul = b[100, 50]  # valor do canal azul na posição (100,50)

# Modificar uma região
g[50:100, 50:100] = 255  # torna verde puro uma região quadrada

# Create a solid red image (B G R)
img_teste = np.full((500, 500, 3), [0, 0, 255], dtype=np.uint8)
#showIMG(imagem_recombinada)

def onlyRED(img):
    b, g, r = cv2.split(img)
    b[:,:] = 0
    g[:,:] = 0
    imagem_recombinada = cv2.merge([b, g, r])
    showIMG(imagem_recombinada)

#onlyRED(img)

def switch(img):
    b, g, r = cv2.split(img)

    v = (b, g, r)

    imagem_recombinada = cv2.merge([v[random.randint(0,100) % 3], v[random.randint(0,100) % 3], v[random.randint(0,100) % 3]])
    showIMG(imagem_recombinada)

#switch(img)

def colorBalance(img, alpha):
    # Converta para float para evitar overflow
    img_float = img.astype(np.float32)
    
    b, g, r = cv2.split(img_float)
    b = np.clip(b * alpha, 0, 255)
    g = np.clip(g * alpha, 0, 255) 
    r = np.clip(r * alpha, 0, 255)
    
    # Converta de volta para uint8
    img_recombinada = cv2.merge([b, g, r]).astype(np.uint8)
    showIMG(img_recombinada)

colorBalance(img, 1.1)




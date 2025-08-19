# Capítulo 3 - Image Processing

Nesse capítulo, trataremos do conjunto de operações e técnicas aplicadas a imagens digitais. É uma etapa de preparação para muitos algoritmos de reconhecimento de objetos, reconstrução 3D e/ou rastreamento de movimento. Nessa perspectiva, começaremos
pelos chamados "operadores de ponto", que manipulam cada pixel de uma imagem, independentemente daqueles ao seu redor.
Depois, passaremos pelos operadores "area-based", nos quais cada novo valor de um pixel depende de um certo número de pontos vizinhos.

Uma imagem colorida é uma matriz 3D (altura × largura × 3 canais). Os 3 canais são: Blue, Green, Red (em OpenCV é BGR, não RGB!)

## Operadores de Ponto

1. Ajuste de Brilho (Brightness)

    Tipo: Transformação linear aditiva

    Fórmula: output = input + beta

    Efeito: Desloca todos os pixels igualmente

    Quando usar: Quando a imagem está muito escura ou muito clara globalmente

2. Ajuste de Contraste (Contrast)

    Tipo: Transformação linear multiplicativa

    Fórmula: output = alpha * input

    Efeito: Expande ou comprime a faixa de valores

    Quando usar: Quando a imagem está "chapada", sem variação tonal

3. Correção Gamma

    Tipo: Transformação não-linear exponencial

    Fórmula: output = input ^ (1/gamma)

    Efeito: Ajusta seletivamente tons médios

    Quando usar: Para corrigir percepção visual ou problemas de iluminação não-linear
    
## Color Balance
Nosso exercício será ajustar a intensidade relativa das cores primárias. Iremos multiplicar cada canal por um fator que altera seu brilho.


###### Renzo Real Machado Filho - 19/05/25.
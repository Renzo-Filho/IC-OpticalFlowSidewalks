# Capítulo 3 - Image Processing

Nesse capítulo, trataremos do conjunto de operações e técnicas aplicadas a imagens digitais. É uma etapa de preparação para muitos algoritmos de reconhecimento de objetos, reconstrução 3D e/ou rastreamento de movimento. Nessa perspectiva, começaremos pelos chamados "operadores de ponto", que manipulam cada pixel de uma imagem, independentemente daqueles ao seu redor. Depois, passaremos pelos operadores "area-based", nos quais cada novo valor de um pixel depende de um certo número de pontos vizinhos.

Um operador genérico é uma função que toma uma ou mais imagens como inputs e produz uma outra imagem de output. Matematicamente,

$$ f(x) = h(g_0(x), \dots, g_n(x)) $$

Já uma imagem digital colorida é representado como uma matriz 3D (altura × largura × 3 canais). Os 3 canais são: Blue, Green, Red (em OpenCV é BGR).

## Operadores de Ponto

#### 1. Ajuste de Brilho (Brightness) e Contraste (Contrast)

Tipo: Transformação linear.
    
Fórmula: $ f(x) = a(x) \cdot g(x) + b(x) $, onde $a$ é dito o contraste e $b$, o brilho.

Efeito: O brilho desloca uniformemente todos os valores de pixel na imagem. Valores positivos clareiam a imagem, valores negativos escurecem. Enquanto isso, o contraste controla a diferença entre tons claros e escuros. Valores > 1 expandem a faixa tonal (aumentam o contraste), valores entre 0 e 1 comprimem a faixa tonal (reduzem o contraste).

Quando usar: Quando a imagem está muito escura ou muito clara globalmente ou quando a imagem está "chapada", i.e, sem muita variação tonal.

#### 2. Correção Gamma

Tipo: Transformação não-linear.

Fórmula: $f(x) = g(x)^{1/\gamma}$.

O que é Gamma ($\gamma$)?: É um parâmetro que define a curvatura da transformação não-linear aplicada aos valores de intensidade. Ele controla como os valores intermediários (tons médios) são mapeados, enquanto preserva os extremos (preto puro e branco puro). Por padrão, usa-se $\gamma \approx 2.2$.

Efeito: Se $\gamma > 1$, escurece os tons médios enquanto preserva pretos e brancos (aumenta contraste em tons escuros). Já $\gamma < 1$, clareia os tons médios enquanto preserva pretos e brancos (aumenta contraste em tons claros).

Quando usar: Para corrigir percepção visual ou problemas de iluminação não-lineares.

#### Ordem das Operações

Contraste/Brilho → Gamma

O Contraste/Brilho são lineares, trabalham no domínio da intensidade enquanto o Gamma é não-linear, trabalha no domínio perceptual. Se fizéssemos gamma primeiro a transformação linear posterior distorceria a curva gamma e os valores seriam re-escalados de forma inadequada, perdendo o controle preciso sobre a correção tonal.


    
## Color Balance
Esse procedimento ajusta a intensidade relativa das cores primárias. A operação é feita por canal (R, G, B). Podemos multiplicá-los por um fator que altera seu brilho ou ainda operar sobre transformações mais complexas, como o mapeamento no espaço de cores XYZ.

###### Espaço de Cores XYZ

É um espaço de cor matematicamente definido, usando coordenadas tridimensionais (X, Y, Z) para descrever todas as cores visíveis ao olho humano.

    X: Representa aproximadamente a sensibilidade ao vermelho

    Y: Representa o brilho luminoso (luminância)

    Z: Representa aproximadamente a sensibilidade ao azul

## Composição e Mascaramento (Compositing and Matting)

Em muitos aplicativos de edição de fotos e de efeitos visuais, queremos inserir/combinar elementos em uma imagem. Esse processo é chamado de **composição** (Smith and Blinn 1996).

Paralelo a isso, também é desejável extrair objetos de imagens, processo comumente chamado de **mascaramento**. Uma máscara define a área de uma imagem que deve ser mantida ou ignorada, permitindo que apenas certas partes da imagem sejam visíveis e sejam integradas com outros elementos (Porter and Duff 1984; Blinn 1994a).

#### Alpha matting 

> É um processo que visa estimar a translucidez de um objeto em uma determinada imagem. O "alpha matting" resultante descreve, em pixels, a quantidade de cores de primeiro e segundo plano que contribuem para a cor da imagem composta. [34]

##### Alpha-Matted Color Image

É uma imagem que além dos 3 canais de cor (RGB), possui um 4º canal intermediário(Alpha - $\alpha$) que representa:

$\alpha$ = 1: Pixel totalmente opaco

$\alpha$ = 0: Pixel totalmente transparente

0 < $\alpha$ < 1: Pixel parcialmente transparente

$$ C = (1-\alpha) B + \alpha F $$




###### Renzo Real Machado Filho.
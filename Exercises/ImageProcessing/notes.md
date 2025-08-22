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
    
## Color Balance
Nosso exercício será ajustar a intensidade relativa das cores primárias. Iremos multiplicar cada canal por um fator que altera seu brilho.


###### Renzo Real Machado Filho - 19/05/25.
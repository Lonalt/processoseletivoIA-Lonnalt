# Relatório do Candidato

**Nome completo:** Luann Alves Pereira De Lima  
**GitHub:** [https://github.com/Lonalt/processoseletivoIA-Lonnalt](https://github.com/Lonalt/processoseletivoIA-Lonnalt)

## 1. Resumo da arquitetura do modelo

O modelo implementado no pipeline é uma rede neural convolucional (CNN) projetada para o dataset MNIST. A arquitetura foca no equilíbrio entre precisão e eficiência para ambientes de Edge AI.

| Etapa | Camada | Configuração | Função |
| :--- | :--- | :--- | :--- |
| Entrada | `Input` | `(28, 28, 1)` | Recebe a imagem normalizada (0 a 1) |
| 1 | `Conv2D` | 32 filtros, 3x3, ReLU | Extração inicial de características |
| 2 | `MaxPooling2D` | 2x2 | Redução de dimensionalidade espacial |
| 3 | `Conv2D` | 64 filtros, 3x3, ReLU | Extração de características de alto nível |
| 4 | `MaxPooling2D` | 2x2 | Segunda redução espacial |
| 5 | `Flatten` | - | Vetorização para a parte densa |
| 6 | `Dense` | 32 unidades, ReLU | Camada de decisão compacta |
| 7 | `Dropout` | taxa 0.3 | Prevenção de overfitting |
| Saída | `Dense` | 10 unidades, Softmax | Classificação das 10 classes (0-9) |

O design utiliza `padding="same"` para preservar informações de borda e garantir uma redução controlada do tensor. A escolha de uma camada densa de 32 unidades visa minimizar o número de parâmetros sem comprometer a acurácia final.

O treinamento foi configurado com o otimizador **Adam**, função de perda `sparse_categorical_crossentropy` e semente aleatória fixa (`seed=7`) para garantir reprodutibilidade.

## 2. Bibliotecas utilizadas

| Biblioteca | Uso no projeto |
| :--- | :--- |
| **TensorFlow / Keras** | Construção da CNN, treinamento e conversão TFLite. |
| **NumPy** | Pré-processamento de matrizes e manipulação de dados numéricos. |
| **OS** | Gerenciamento de arquivos e medição de tamanho em disco. |

## 3. Técnica de otimização do modelo

A otimização foi realizada via **Dynamic Range Quantization (DRQ)** através do `TFLiteConverter`. Esta técnica quantiza os pesos do modelo para 8 bits durante o armazenamento, enquanto as ativações permanecem em ponto flutuante durante a execução, sendo convertidas dinamicamente. 

Essa abordagem foi escolhida por oferecer uma redução significativa de tamanho (aprox. 4x) sem a necessidade de um conjunto de dados de calibração complexo, mantendo a compatibilidade com diversos interpretadores de borda.

## 4. Resultados obtidos

Os resultados validam a eficácia da arquitetura e do processo de compressão:

| Indicador | Resultado |
| :--- | :--- |
| **Parâmetros Treináveis** | 119.530 |
| **Acurácia Keras (Teste)** | 98,56% |
| **Acurácia TFLite (Teste)** | 98,56% |
| **Delta de Acurácia** | 0.00 pp |
| **Tamanho model.h5** | 494,6 KB |
| **Tamanho model.tflite** | 124,5 KB |
| **Redução de Tamanho** | **74,8%** |

## 5. Comentários adicionais

A pipeline demonstrou que a transição de um modelo de quase 500 KB para um de apenas 124 KB foi realizada com **perda zero de acurácia** (mantendo 98,56%). Isso confirma que o modelo original não estava excessivamente comprimido, permitindo que a quantização removesse redundâncias de precisão nos pesos sem afetar a inteligência da rede.

O uso de `include_optimizer=False` no salvamento do arquivo `.h5` também contribuiu para um modelo inicial mais leve, focado estritamente na tarefa de inferência.

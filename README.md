# 📊 Dashboard de Performance: Desafio IA
**Desenvolvedor:** Luann Alves Pereira De Lima  
**Repositório:** [Lonalt/processoseletivoIA-Lonnalt](https://github.com/Lonalt/processoseletivoIA-Lonnalt)

---

## ⚡ Sumário de Eficiência
O projeto alcançou um equilíbrio otimizado entre peso computacional e precisão preditiva, ideal para implantação em hardware limitado.

| Métrica | Status | Resultado |
| :--- | :--- | :--- |
| **Acurácia Final** | 🎯 Excelente | **98,56%** |
| **Integridade de Conversão** | 💎 Perfeita | **0.00 pp de perda** |
| **Taxa de Compressão** | 📉 Alta | **-74,8%** |
| **Complexidade** | ⚙️ Leve | **119.530 parâmetros** |

---

## 🏗️ Engenharia do Modelo (CNN)
A arquitetura foi construída seguindo uma progressão lógica de extração de padrões, projetada para ser robusta contra ruídos sem inflar o uso de memória.

### Fluxo de Dados e Dimensões
1.  **Recepção**: Tensor de entrada `(28, 28, 1)` normalizado em ponto flutuante.
2.  **Extração Espacial**: 
    *   `Conv2D (32)` -> `MaxPooling`: Capta bordas fundamentais.
    *   `Conv2D (64)` -> `MaxPooling`: Reconhece formas complexas (curvas e loops dos números).
3.  **Redução Logística**: 
    *   A camada `Dense` foi limitada a **32 neurônios**. 
    *   **Decisão Técnica**: Esta escolha cortou drasticamente o número de pesos (apenas 100.384 parâmetros nesta camada) em comparação a uma camada de 64, mantendo a capacidade de generalização necessária para o MNIST.
4.  **Regularização**: Inclusão de `Dropout(0.3)` para garantir que a rede não memorize os dados de treino.

---

## 💎 Estratégia de Otimização: DRQ
Para a entrega de borda, foi aplicada a **Dynamic Range Quantization** no `optimize_model.py`.

*   **Implementação**: Conversão do modelo Keras para o formato FlatBuffer do TFLite utilizando `tf.lite.Optimize.DEFAULT`.
*   **Vantagem Técnica**: Reduz os pesos de 32-bit (float) para 8-bit (int), diminuindo o footprint de armazenamento sem exigir um dataset representativo para calibração de ativações.
*   **Validação Crucial**: O script de otimização não apenas gera o arquivo, mas instanciao o `tf.lite.Interpreter` para garantir que a saída `(1, 10)` seja numericamente estável e precisa.

---

## 📈 Análise Comparativa de Recursos

```text
Tamanho em Disco (KB)
[####################] 494.6 KB (Original .h5)
[#####               ] 124.5 KB (Otimizado .tflite)
                       ^-- Redução de 370.1 KB!
```

**Logs de Treinamento (Acurácia por Época):**
*   **E01**: 86.8%
*   **E02**: 94.8%
*   **E03**: 95.9%
*   **E04**: 96.7% (Final de Treino)
*   **Teste**: **98.56%** (Generalização superior)

---

## 🚀 Conclusão de Engenharia
O modelo está pronto para produção em sistemas embarcados. A economia de **74,8%** no espaço em disco permite que o binário seja integrado em aplicações mobile ou microcontroladores com folga de recursos, enquanto a manutenção da acurácia em **98,56%** garante a confiabilidade do sistema de visão computacional.

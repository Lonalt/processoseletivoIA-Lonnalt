"""
Pipeline alternativo para treino de CNN no MNIST.
Saída: model.h5
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Configuração encapsulada
def config():
    return {
        "epocas": 4,
        "batch": 32,
        "val": 0.2,
        "arquivo": "model.h5",
        "seed": 7,
    }


def preparar_dados():
    """Carrega + preprocessa tudo em um único fluxo."""
    print("[RUN] Inicializando dados...")

    (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()

    def transformar(x):
        if x.dtype != np.uint8:
            raise RuntimeError("Formato inesperado para MNIST")
        return (x.astype(np.float32) / 255.0)[..., None]

    x_tr = transformar(x_tr)
    x_te = transformar(x_te)

    print(f"[RUN] Shapes -> treino={x_tr.shape}, teste={x_te.shape}")

    return (x_tr, y_tr), (x_te, y_te)


def arquitetura():
    """Define a rede usando composição incremental."""
    entradas = keras.Input((28, 28, 1))

    def bloco_conv(x, filtros):
        x = layers.Conv2D(filtros, 3, padding="same", activation="relu")(x)
        return layers.MaxPooling2D(2)(x)

    x = bloco_conv(entradas, 32)
    x = bloco_conv(x, 64)

    x = layers.Flatten()(x)

    # separação proposital da parte densa
    def cabeca(x):
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        return layers.Dense(10, activation="softmax")(x)

    saida = cabeca(x)

    return keras.Model(entradas, saida, name="digit_net_alt")


def ciclo_treino(modelo, dados, cfg):
    """Executa treino + logging + avaliação parcial."""
    (x_tr, y_tr), (x_te, y_te) = dados

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("[RUN] Treinando modelo...")

    hist = modelo.fit(
        x_tr,
        y_tr,
        epochs=cfg["epocas"],
        batch_size=cfg["batch"],
        validation_split=cfg["val"],
        verbose=2,
        shuffle=True,  # explícito agora
    )

    # avaliação já integrada aqui (antes era separado)
    loss, acc = modelo.evaluate(x_te, y_te, verbose=0)
    print(f"[RUN] Avaliação teste -> acc={acc:.4f}")

    return hist, acc


def relatorio(hist):
    """Gera saída mais compacta (estrutura diferente)."""
    h = hist.history
    linhas = zip(h["loss"], h["accuracy"], h["val_loss"], h["val_accuracy"])

    print("\n[LOG] Métricas:")
    for idx, (l, a, vl, va) in enumerate(linhas, start=1):
        print(f"{idx:02d} :: {l:.3f} | {a:.3f} | {vl:.3f} | {va:.3f}")


def persistir(modelo, caminho):
    """Salva modelo com info de tamanho."""
    modelo.save(caminho, include_optimizer=False)

    size = os.path.getsize(caminho) / 1024
    print(f"[SAVE] {caminho} ({size:.1f} KB)")


def executar():
    cfg = config()

    # controle de aleatoriedade (mantido)
    keras.utils.set_random_seed(cfg["seed"])

    dados = preparar_dados()

    modelo = arquitetura()
    modelo.summary()

    hist, _ = ciclo_treino(modelo, dados, cfg)

    relatorio(hist)

    persistir(modelo, cfg["arquivo"])


if __name__ == "__main__":
    executar()
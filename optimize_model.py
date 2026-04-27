"""
Exportacao e validacao de modelo para TFLite (quantizacao dinamica)
Entrada: model.h5 / model.keras
Saida: model.tflite
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

def parametros():
    return {
        "entrada_modelo": "model.h5",
        "saida_tflite": "model.tflite",
        "amostras": 10000,
    }


def carregar_e_converter(cfg):
    """Carrega modelo e já retorna versão TFLite."""
    print(f"[PIPE] Lendo modelo: {cfg['entrada_modelo']}")

    modelo = keras.models.load_model(cfg["entrada_modelo"], compile=False)

    print("[PIPE] Convertendo para TFLite (DRQ)...")

    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_bytes = conv.convert()

    with open(cfg["saida_tflite"], "wb") as arq:
        arq.write(tflite_bytes)

    print(f"[PIPE] Arquivo gerado: {cfg['saida_tflite']}")

    return modelo

def preparar_amostras(qtd):
    """Carrega MNIST e retorna subconjunto pronto."""
    (_, _), (x, y) = keras.datasets.mnist.load_data()

    x = (x.astype(np.float32) / 255.0)[..., None]

    return x[:qtd], y[:qtd]

def rodar_tflite(caminho, imagens):
    """Executa inferência via interpreter."""
    interp = tf.lite.Interpreter(model_path=caminho)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]["index"]
    out = interp.get_output_details()[0]["index"]

    resultados = []

    for idx, img in enumerate(imagens):
        interp.set_tensor(inp, img[None, ...])
        interp.invoke()

        saida = interp.get_tensor(out)

        if idx == 0:
            assert saida.shape[-1] == 10
            assert np.isfinite(saida).all()

        resultados.append(np.argmax(saida))

    return np.array(resultados)


def comparar(modelo, imagens, rotulos, preds_tflite):
    """Compara TFLite vs Keras."""
    acc_tflite = (preds_tflite == rotulos).mean() * 100
    print(f"[CHECK] TFLite acc: {acc_tflite:.2f}%")

    if modelo is not None:
        preds_keras = np.argmax(modelo.predict(imagens, verbose=0), axis=1)
        acc_keras = (preds_keras == rotulos).mean() * 100

        print(f"[CHECK] Keras acc: {acc_keras:.2f}%")
        print(f"[CHECK] Dif: {acc_tflite - acc_keras:+.2f} pp")

def estatisticas(antes, depois):
    """Mostra redução de tamanho."""
    s1 = os.path.getsize(antes) / 1024
    s2 = os.path.getsize(depois) / 1024

    delta = 100 * (1 - (s2 / s1))

    print("\n[SIZE]")
    print(f"{antes}: {s1:.1f} KB")
    print(f"{depois}: {s2:.1f} KB")
    print(f"Reducao: {delta:.1f}%")


def fluxo():
    cfg = parametros()

    modelo = carregar_e_converter(cfg)

    imgs, lbls = preparar_amostras(cfg["amostras"])

    preds = rodar_tflite(cfg["saida_tflite"], imgs)

    comparar(modelo, imgs, lbls, preds)

    estatisticas(cfg["entrada_modelo"], cfg["saida_tflite"])


if __name__ == "__main__":
    fluxo()